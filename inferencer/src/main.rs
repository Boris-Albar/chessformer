extern crate mcts;
extern crate chess;
extern crate intmap;
extern crate ndarray;
extern crate ndarray_npy;
extern crate rand;
extern crate rand_distr;
extern crate shakmaty;
extern crate shakmaty_syzygy;
extern crate parking_lot;
extern crate tensorrt_rs;
extern crate cached;
extern crate glob;

use rand::prelude::*;
use rand::distributions::WeightedIndex;

extern crate clap;
use clap::{Arg, App};
use std::sync::Arc;

use ndarray::Array1;
use ndarray_npy::NpzWriter;

use std::fs::File;
use std::io::Write;
use std::time::SystemTime;
use std::thread;
use std::time::{Duration, Instant};

use mcts::*;
use mcts::tree_policy::*;
use mcts::transposition_table::*;

/*extern crate nix;
use nix::unistd::{fork, ForkResult};*/

mod chess_game;
mod material_mcts;
mod fast_math;
mod barrier;
mod semarrier;
mod trt_inferer;
mod nn_trt_mcts;

use crate::trt_inferer::TrtInferer;

fn simulate_game(thread_number: u64,
        inferer: Arc<TrtInferer>,
        game_path: &str,
        runner_name: &str,
        mcts_threads: usize,
        mcts_moves: u64,
        use_time: bool,
        virtual_loss: i64,
        visits_before_expansion: u64,
        stochastic_moves: bool,
        stochastic_temperature: f64,
        policy_temperature: f64,
        dirichlet_epsilon: f64,
        dirichlet_alpha: f64,
        syzygy_path: &str,
        syzygy_stopping: bool,
        ) {

    let mut game = chess_game::ChessGame::new_game(syzygy_path.to_string());

    let mut game_status = game.game_status();
    let time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let game_number = time.as_secs() * 1000 + time.subsec_nanos() as u64 / 1_000_000;

    let mut npz = NpzWriter::new(File::create(game_path.to_owned() + "/game_moves_" + runner_name + "_" + &thread_number.to_string() + "_" + &game_number.to_string() + ".npz").expect("Unable to create npz file"));

    let mut vector_fen = Vec::<String>::with_capacity(256);
    let mut probability_moves = Array1::<i64>::zeros(4228);
    let mut num_move = 0;

    let start_time = Instant::now();

    while game_status == chess_game::ChessGameStatus::Ongoing {
        // Generate a MCTS Manager
        let mut mcts = MCTSManager::new(game.clone(),
            nn_trt_mcts::ChessformerTrtMCTS::default(virtual_loss, visits_before_expansion),
            nn_trt_mcts::NNTrtEvaluator::load_from_pointer(inferer.clone(), policy_temperature, dirichlet_epsilon, dirichlet_alpha, game.current_player()),
            AlphaGoPolicy::new(5.0, 1.0, false),
            ApproxTable::new(1024));
        //let mut mcts = MCTSManager::new(game.clone(), material_mcts::ChessMaterialMCTS, material_mcts::MaterialEvaluator, AlphaGoPolicy::new(5.0, 1.0, false), ApproxTable::new(1024));

        // reset probability move array
        probability_moves.iter_mut().for_each(|m| *m = 0);

        if use_time == true {
            mcts.playout_parallel_for(Duration::from_millis(mcts_moves), mcts_threads);
        } else {
            if mcts_threads == 1 {
                mcts.playout_n(mcts_moves);
            } else {
                mcts.playout_n_parallel(mcts_moves as u32, mcts_threads);
            }
        }

        /* select a move in a stochastic way or take the first move */
        let mut selected_move = 0;
        if stochastic_moves == true {
            let vec_visits: Vec<f64> = mcts.get_root_moves_info().iter().map(|x| ((x.visits() as f64).powf(1.0 / stochastic_temperature))).collect();
            let dist = WeightedIndex::new(&vec_visits).unwrap();
            selected_move = dist.sample(&mut rand::thread_rng());
        }

        for (i, x) in mcts.get_root_moves_info().iter().enumerate() {

            let var_move = x.get_move();
            let source = var_move.get_source().to_index() as i64;
            let dest = var_move.get_dest().to_index() as i64;

            if let Some(piece) = var_move.get_promotion() {
                let is_white_promoting = (source < 20) as i64;

                match piece { // seventh rank
                    chess::Piece::Knight => probability_moves[(4096 + (66 * is_white_promoting) + 0 + ((source % 8) * 3 - 1) + ((dest - source) - 8)) as usize] = x.visits() as i64,
                    chess::Piece::Bishop => probability_moves[(4096 + (66 * is_white_promoting) + 22 + ((source % 8) * 3 - 1) + ((dest - source) - 8)) as usize] = x.visits() as i64,
                    chess::Piece::Rook => probability_moves[(4096 + (66 * is_white_promoting) + 44 + ((source % 8) * 3 - 1) + ((dest - source) - 8)) as usize] = x.visits() as i64,
                    _ => probability_moves[(source * 64 + dest) as usize] = x.visits() as i64,
                }
            } else {
                probability_moves[(source * 64 + dest) as usize] = x.visits() as i64;
            }

            /* make the move on the game as selected before */
            if i == selected_move {
                game.make_move(var_move);
            }

        }

        vector_fen.push(game.board_string() + "--" + &game.repetition_number().to_string() + "--" + &game.fifty_rules_counter().to_string());
        npz.add_array(num_move.to_string() , &probability_moves).expect("Unable to array data!");

        game_status = game.game_status();
        num_move += 1;

        /* early stopping if we reached a syzygy table position */
        if syzygy_stopping == true {
            let syzygy_eval = game.get_syzygy_wdl();

            if game.current_player() == chess::Color::White {
                match syzygy_eval {
                    x if x == -1.0 => game_status = chess_game::ChessGameStatus::BlackWon,
                    x if x == 0.0 => game_status = chess_game::ChessGameStatus::DrawBy50Rule,
                    x if x == 1.0 => game_status = chess_game::ChessGameStatus::WhiteWon,
                    _ => (),
                }
            } else {
                match syzygy_eval {
                    x if x == -1.0 => game_status = chess_game::ChessGameStatus::WhiteWon,
                    x if x == 0.0 => game_status = chess_game::ChessGameStatus::DrawBy50Rule,
                    x if x == 1.0 => game_status = chess_game::ChessGameStatus::BlackWon,
                    _ => (),
                }
            }

        }
    }

    let mut file_boards = File::create(game_path.to_owned() + "/game_boards_" + runner_name +  "_" + &thread_number.to_string() + "_" + &game_number.to_string() + ".txt").expect("Unable to create board file");
    for i in &vector_fen{
        write!(&mut file_boards, "{:?}\n", *i).expect("Unable to write data!");
    }
    write!(&mut file_boards, "{:?}\n", game_status).expect("Unable to write data!");

    let total_time = start_time.elapsed();
    println!("Game generated : {:?}, {:?} samples, {:?} secs", game_status, game.num_moves(), total_time);

}

fn thread_loop(thread_number: u64,
        inferer: Arc<TrtInferer>,
        game_path: String,
        runner_name: String,
        mcts_threads: usize,
        mcts_moves: u64,
        use_time: bool,
        stochastic_moves: bool,
        stochastic_temperature: f64,
        policy_temperature: f64,
        dirichlet_epsilon: f64,
        dirichlet_alpha: f64,
        syzygy_path: String,
        syzygy_stopping: bool,
        virtual_loss: i64,
        visits_before_expansion: u64,
        max_games: usize,
        ) {

     let mut counter = 0;

     while (max_games == 0) || (counter < max_games) {

        simulate_game(thread_number, inferer.clone(), &game_path, &runner_name, mcts_threads, mcts_moves, use_time,
            virtual_loss, visits_before_expansion, stochastic_moves,
            stochastic_temperature, policy_temperature, dirichlet_epsilon, dirichlet_alpha,
            &syzygy_path, syzygy_stopping);
    }
}

fn main() {

    let matches = App::new("Chessformer Inferer")
        .version("0.3.0")
        .author("Boris Albar <boris.a@sleipnir.fr>")
        .arg(Arg::with_name("onnx_path")
                 .long("onnx_path")
                 .takes_value(true)
                 .help("Path of onnx files"))
        .arg(Arg::with_name("games_path")
                 .long("games_path")
                 .takes_value(true)
                 .help("Path of games files"))
        .arg(Arg::with_name("parallel_games")
                 .long("parallel_games")
                 .takes_value(true)
                 .help("Number of parallel games"))
        .arg(Arg::with_name("mcts_threads")
                 .long("mcts_threads")
                 .takes_value(true)
                 .help("Number of MCTS threads"))
        .arg(Arg::with_name("mcts_moves")
                 .long("mcts_moves")
                 .takes_value(true)
                 .help("Number of MCTS moves"))
        .arg(Arg::with_name("use_time")
                 .long("use_time")
                 .takes_value(false)
                 .help("In case of true, mcts_moves corresponds to a time in ms"))
        .arg(Arg::with_name("use_stochastic_moves")
                 .long("use_stochastic_moves")
                 .takes_value(false)
                 .help("In case of true, sample a random move proportionnal to the probability"))
        .arg(Arg::with_name("dirichlet_noise")
                 .long("dirichlet_alpha")
                 .takes_value(true)
                 .help("Value alpha of the Dirichlet distribution"))
        .arg(Arg::with_name("policy_noise_amount")
                 .long("dirichlet_epsilon")
                 .takes_value(true)
                 .help("Amount of noise to add in the policy"))
        .arg(Arg::with_name("policy_temperature")
                 .long("policy_temperature")
                 .takes_value(true)
                 .help("Temperature to apply on the probabilities vector of the NN"))
        .arg(Arg::with_name("syzygy_path")
                 .long("syzygy_path")
                 .takes_value(true)
                 .help("Path to the syzygy table"))
        .arg(Arg::with_name("syzygy_stopping")
                 .long("syzygy_stopping")
                 .takes_value(false)
                 .help("Stop when a Syzygy table position is reached"))
        .arg(Arg::with_name("enable_cache")
                 .long("enable_cache")
                 .takes_value(false)
                 .help("Enable caching mechanism before inference"))
        .arg(Arg::with_name("batch_size")
                 .long("batch_size")
                 .takes_value(true)
                 .help("Batch size to use for inference"))
        .arg(Arg::with_name("timeout")
                 .long("timeout")
                 .takes_value(true)
                 .help("Timeout to release lock for inference"))
        .arg(Arg::with_name("cache_size")
                 .long("cache_size")
                 .takes_value(true)
                 .help("Cache size to use for inference"))
        .arg(Arg::with_name("runner_name")
                 .long("runner_name")
                 .takes_value(true)
                 .help("A runner name for multiple instances"))
        .arg(Arg::with_name("max_games")
                 .long("max_games")
                 .takes_value(true)
                 .help("A maximum number of games per threads (0 for unlimited)"))
        .arg(Arg::with_name("gpu_number")
                 .long("gpu_number")
                 .takes_value(true)
                 .help("Number of the gpu to use"))
        .get_matches();

    let runner_name = Arc::new(matches.value_of("runner_name").unwrap_or("gpu0"));
    let game_path = Arc::new(matches.value_of("games_path").unwrap_or("games/"));
    let onnx_path = matches.value_of("onnx_path").unwrap_or("../trainer/weights/");
    let parallel_games: u64 = matches.value_of("parallel_games").unwrap_or("1").parse().unwrap_or(1);
    let virtual_loss: i64 = matches.value_of("virtual_loss").unwrap_or("3").parse().unwrap_or(3);
    let visits_before_expansion: u64 = matches.value_of("visits_before_expansion").unwrap_or("20").parse().unwrap_or(20);
    let mcts_threads: usize = matches.value_of("mcts_threads").unwrap_or("4").parse().unwrap_or(4);
    let mcts_moves: u64 = matches.value_of("mcts_moves").unwrap_or("800").parse().unwrap_or(800);
    let use_time = matches.is_present("use_time");
    let use_stochastic_moves = matches.is_present("use_stochastic_moves");
    let stochastic_temperature: f64 = matches.value_of("stochastic_temperature").unwrap_or("1.0").parse().unwrap_or(1.0);
    let dirichlet_alpha: f64 = matches.value_of("dirichlet_alpha").unwrap_or("0.3").parse().unwrap_or(0.3);
    let dirichlet_epsilon: f64 = matches.value_of("dirichlet_epsilon").unwrap_or("0.25").parse().unwrap_or(0.25);
    let policy_temperature: f64 = matches.value_of("policy_temperature").unwrap_or("0.75").parse().unwrap_or(0.75);
    let syzygy_path = Arc::new(matches.value_of("syzygy_path").unwrap_or(""));
    let syzygy_stopping = matches.is_present("syzygy_stopping");
    let batch_size: usize = matches.value_of("batch_size").unwrap_or("1").parse().unwrap_or(1);
    let timeout: u64 = matches.value_of("timeout").unwrap_or("100").parse().unwrap_or(20);
    let enable_cache = matches.is_present("enable_cache");
    let cache_size: usize = matches.value_of("cache_size").unwrap_or("2048").parse().unwrap_or(2048);
    let max_games: usize = matches.value_of("max_games").unwrap_or("0").parse().unwrap_or(0) / parallel_games;
    let gpu_number: u64 = matches.value_of("gpu_number").unwrap_or("0").parse().unwrap_or(0);

    assert!(batch_size <= ((parallel_games as usize) * mcts_threads),
        "Batch size cannot be more than mcts_threads multiplied by parallel games.");


    let mut threads = Vec::new();
    let trt_inferer_engine = TrtInferer::new(onnx_path, gpu_number, cache_size, batch_size, timeout, enable_cache);
    let trt_inferer_ptr = Arc::new(trt_inferer_engine);

    for i in 0..parallel_games {
        let game_var = game_path.to_string();
        let syzygy_path_var = syzygy_path.to_string();
        let runner_name_var = runner_name.to_string();
        let trt_clone = trt_inferer_ptr.clone();
        threads.push(thread::spawn(move || {
                thread_loop(i, trt_clone , game_var, runner_name_var, mcts_threads, mcts_moves, use_time,
                    use_stochastic_moves, stochastic_temperature, policy_temperature,
                    dirichlet_epsilon, dirichlet_alpha, syzygy_path_var, syzygy_stopping,
                    virtual_loss, visits_before_expansion, max_games);
            }));
    }

    for thrd in threads {
        thrd.join().expect("Game thread has panicked");
    }

    /*for j in 0..num_gpus {
        match unsafe{fork()} {
            Ok(ForkResult::Parent { child, .. }) => (),
            Ok(ForkResult::Child) => {
                let mut threads = Vec::new();
                let trt_inferer_engine = TrtInferer::new(onnx_path, j, cache_size, batch_size, timeout, enable_cache);
                let trt_inferer_ptr = Arc::new(trt_inferer_engine);

                for i in 0..parallel_games {
                    let game_var = game_path.to_string();
                    let syzygy_path_var = syzygy_path.to_string();
                    let trt_clone = trt_inferer_ptr.clone();
                    threads.push(thread::spawn(move || {
                            thread_loop(((i+j) * (i+j+1))/2 + j, trt_clone , game_var, mcts_threads, mcts_moves, use_time,
                                use_stochastic_moves, stochastic_temperature, policy_temperature,
                                dirichlet_epsilon, dirichlet_alpha, syzygy_path_var, syzygy_stopping,
                                virtual_loss, visits_before_expansion);
                        }));
                }

                for thrd in threads {
                    thrd.join().expect("Game thread has panicked");
                }

            },
            Err(_) => println!("Fork failed!"),
        }
    }

    loop {
        thread::sleep(Duration::from_millis(10000));
    }*/
}
