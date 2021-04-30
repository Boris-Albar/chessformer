use ndarray::{Array1, Array2, Array3, Axis, concatenate};

use std::time::Duration;
use std::env;
use std::sync::{atomic};
use parking_lot::{RwLock};
use crate::semarrier::Semarrier;
use crate::barrier::Barrier;

use glob::glob;

use tensorrt_rs::builder::{Builder, NetworkBuildFlags};
use tensorrt_rs::context::{Context, ExecuteInput};
use tensorrt_rs::data_size::GB;
use tensorrt_rs::dims::Dims3;
use tensorrt_rs::engine::Engine;
use tensorrt_rs::onnx::{OnnxFile, OnnxParser};
use tensorrt_rs::runtime::Logger;
use std::path::PathBuf;

use cached::{Cached, SizedCache};

struct BatchData {
    array: Vec<Array3<f32>>,
    hashes: Vec<u64>,
}

impl BatchData {

    pub fn new(batch_size: usize) -> BatchData {
        BatchData {
            array: Vec::<Array3<f32>>::with_capacity(batch_size),
            hashes: Vec::<u64>::with_capacity(batch_size),
        }
    }

    #[inline]
    pub fn push(&mut self, input_board: Array3<f32>, board_hash: u64) {
        self.array.push(input_board);
        self.hashes.push(board_hash);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.array.len()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.array.clear();
        self.hashes.clear();
    }
}

pub struct TrtInferer {
    logger: Logger,
    engine: Engine,
    context: Context,
    cache: RwLock<SizedCache<u64, Array1<f32>>>,
    batch_size: usize,
    entry_semaphore: Semarrier,
    comput_barrier: Barrier,
    shared_batch: RwLock<BatchData>,
    shared_result: RwLock<Array2<f32>>,
    timeout_hit: atomic::AtomicUsize,
    enable_cache: bool,
}

fn get_last_weights(onnx_path: &str) -> PathBuf {
    // Get the last weights
    let name_files: Vec<_> = glob(&(onnx_path.to_owned() + "/*.onnx")).expect("Failed to read glob pattern").collect();
    let mut path_names: Vec<_> = name_files.iter().map(|a| a.as_ref().unwrap()).collect();
    path_names.sort_unstable_by(|a, b| b.cmp(&a)); // reverse order

    path_names[0].to_path_buf()
}

fn create_trt_engine(
    logger: &Logger,
    file: OnnxFile,
    batch_size: i32,
    workspace_size: usize,
    enable_fp16: bool
) -> Engine {
    let builder = Builder::new(&logger);
    builder.set_max_workspace_size(workspace_size);
    let network = builder.create_network_v2(NetworkBuildFlags::EXPLICIT_BATCH);
    let verbosity = 7;

    builder.set_max_batch_size(batch_size);
    builder.set_max_workspace_size(workspace_size);
    if enable_fp16 == true {
        builder.set_fp16_mode(true);
    }

    let parser = OnnxParser::new(&network, &logger);
    parser.parse_from_file(&file, verbosity).unwrap();

    // put board dimension here
    let dim = Dims3::new(batch_size, 65, 15);
    network.get_input(0).set_dimensions(dim);
    builder.build_cuda_engine(&network)
}

impl TrtInferer {

    pub fn new(onnx_path: &str, gpu_number: u64, cache_size: usize, batch_size: usize, timeout: u64, enable_cache: bool, enable_fp16: bool) -> TrtInferer {

        /* Set up the gpu to use */
        env::set_var("CUDA_VISIBLE_DEVICES", gpu_number.to_string());

        let last_weights = get_last_weights(&onnx_path);
        let logger = Logger::new();
        let file = OnnxFile::new(&last_weights).expect("Failed to load onnx weights");
        let engine = create_trt_engine(&logger, file, batch_size as i32, 2 * GB, enable_fp16);

        let context = engine.create_execution_context();

        TrtInferer {
            logger: logger,
            engine: engine,
            context: context,
            cache: RwLock::new(SizedCache::<u64, Array1<f32>>::with_size(cache_size)),
            batch_size: batch_size,
            entry_semaphore: Semarrier::new(batch_size, Duration::from_millis(timeout)),
            comput_barrier: Barrier::new(batch_size),
            shared_batch: RwLock::new(BatchData::new(batch_size)),
            shared_result: RwLock::new(Array2::<f32>::zeros((batch_size, 4229))),
            timeout_hit: atomic::AtomicUsize::new(0),
            enable_cache: enable_cache,
        }
    }

    pub fn get_batch_nn_evaluation(&self, input_board: Array3<f32>, board_hash: u64) -> Array1<f32> {

        if self.enable_cache == true {
            /* read the cache in case it already is inside */
            let cache_read = self.cache.read();

            // ugly convert to mutable for parallelization
            let cache_ptr = &*cache_read as *const SizedCache::<u64, Array1<f32>> as *const ();
            let mut_cache_ptr = cache_ptr as *mut SizedCache::<u64, Array1<f32>>;
            let cache_mut = unsafe { &mut *mut_cache_ptr};

            let result = cache_mut.cache_get(&board_hash);

            if let Some(cached_array) = result {
                return cached_array.clone();
            }
            drop(cache_read);
        }

        let leader = self.entry_semaphore.acquire(); // no more than batch_size threads inside the critic area

        //println!("Semaphore acquired with batch size {}", self.entry_semaphore.get_optim_threads());
        let mut comput_barrier = self.comput_barrier.clone();
        comput_barrier.set_max(self.entry_semaphore.get_optim_threads());

        // take the context mutex for shared memory and add the element to the batch
        let mut shared_batch_mut = self.shared_batch.write();

        shared_batch_mut.push(input_board, board_hash);
        let index = shared_batch_mut.len() - 1;

        drop(shared_batch_mut); // release the vec mutability

        //println!("Index {} arrived, is leader {}", index, leader);

        comput_barrier.wait(); // wait that all threads have deposited their data

        // if batch is complete, notify the group
        if index != 0 {
            //println!("Index {} waiting for computation", index);
            comput_barrier.wait();
            //println!("Index {} released computation", index);
        } else {
            let mut shared_batch_mut = self.shared_batch.write();
            if shared_batch_mut.len() != self.batch_size {
                self.timeout_hit.fetch_add(1, atomic::Ordering::SeqCst);
            }

            let mut batch_view = Vec::with_capacity(self.batch_size);
            for x in shared_batch_mut.array.iter() {
                batch_view.push(x.view());
             }
            let mut batch = concatenate(Axis(0), batch_view.as_slice()).unwrap();

            let mut output = self.shared_result.write();
            let outputs = vec![ExecuteInput::Float(&mut output)];

            self.context.execute(ExecuteInput::Float(&mut batch), outputs).unwrap();

            // put all elements into cache
            if self.enable_cache == true {
                let cache_write = self.cache.write();

                // ugly convert to mutable
                let cache_ptr = &*cache_write as *const SizedCache::<u64, Array1<f32>> as *const ();
                let mut_cache_ptr = cache_ptr as *mut SizedCache::<u64, Array1<f32>>;
                let cache_mut = unsafe { &mut *mut_cache_ptr};

                for i in 0..shared_batch_mut.len() {
                    cache_mut.cache_set(shared_batch_mut.hashes[i], output.index_axis(Axis(0), index).to_owned());
                }
                drop(cache_write);
            }

            // release the output lock
            drop(output);

            // reset the batch
            shared_batch_mut.clear();
            drop(shared_batch_mut); // release the vec mutability
            comput_barrier.wait(); // unlock the barrier
            //println!("Index 0 outside shared batch");
        }

        self.entry_semaphore.release();
        //println!("Index {} released", index);

        let reader_refcell = self.shared_result.read();
        //println!("Index {} in result read", index);
        let result = reader_refcell.index_axis(Axis(0), index).to_owned();
        drop(reader_refcell);
        //println!("Index {} ouside result read", index);

        result
    }

    /*pub fn get_nn_evaluation(&self, mut input_board: Array3<f32>, board_hash: u64) -> Array1<f32> {

        /* read the cache in case it already is inside */
        let cache_read = self.cache.read();
        // ugly convert to mutable for parallelization
        let cache_ptr = &*cache_read as *const SizedCache::<u64, Array1<f32>> as *const ();
        let mut_cache_ptr = cache_ptr as *mut SizedCache::<u64, Array1<f32>>;
        let cache_mut = unsafe { &mut *mut_cache_ptr};

        let result = cache_mut.cache_get(&board_hash);

        if let Some(cached_array) = result {
            return cached_array.clone();
        }
        drop(cache_read);


        let mut output = Array1::<f32>::zeros(4229);
        let outputs = vec![ExecuteInput::Float(&mut output)];

        self.context.execute(ExecuteInput::Float(&mut input_board), outputs).unwrap();
        cache.cache_set(board_hash, output.clone());
        drop(guard); // release the mutex

        output
    }*/
}
