extern crate parking_lot;
use parking_lot::{Condvar, Mutex};

use std::cmp;
use std::time::Duration;
use std::sync::atomic;

use std::ops::Drop;
/*use std::sync::{Condvar, Mutex};*/

/// A barrier which enables multiple threads to synchronize the beginning of some computation.
pub struct Semarrier {
    lock: Mutex<usize>,
    cvar: Condvar,
    max_threads: usize,
    optim_threads: atomic::AtomicUsize,
    inside_count: atomic::AtomicUsize,
    is_blocked: atomic::AtomicBool,
    timeout: Duration,
}

/// An RAII guard which will release a resource acquired from a semaphore when
/// dropped.
pub struct SemarrierGuard<'a> {
    sem: &'a Semarrier,
}

impl Semarrier {
    /// Creates a new semaphore with the initial count specified.
    ///
    /// The count specified can be thought of as a number of resources, and a
    /// call to `acquire` or `access` will block until at least one resource is
    /// available. It is valid to initialize a semaphore with a negative count.
    pub fn new(count: usize, timeout: Duration) -> Semarrier {
        Semarrier {
            lock: Mutex::new(0),
            cvar: Condvar::new(),
            max_threads: count,
            optim_threads: atomic::AtomicUsize::new(count),
            inside_count: atomic::AtomicUsize::new(0),
            is_blocked: atomic::AtomicBool::new(false),
            timeout: timeout,
        }
    }

    /// Acquires a resource of this semaphore, blocking the current thread until
    /// it can do so.
    ///
    /// This method will block until the internal count of the semaphore is at
    /// least 1.
    pub fn acquire(&self) -> bool {
        let mut count = self.lock.lock();
        let mut has_timeouted = false;

        *count += 1;
        while (*count < self.optim_threads.load(atomic::Ordering::SeqCst)) ||
                (has_timeouted == true) ||
                (self.inside_count.load(atomic::Ordering::SeqCst) >= self.max_threads) ||
                (self.is_blocked.load(atomic::Ordering::SeqCst) == true) {
            has_timeouted = self.cvar.wait_for(&mut count, self.timeout).timed_out();
            if (has_timeouted == true) && (self.inside_count.compare_and_swap(0, 1, atomic::Ordering::SeqCst) == 0) {
                // first timeout, release the thread
                break;
            }
        }

        if has_timeouted == false {
            let index = self.inside_count.fetch_add(1, atomic::Ordering::SeqCst);
            if index == 0 { // first thread to be release
                let mut num_threads: usize = 1;
                while (num_threads < self.max_threads) && self.cvar.notify_one() {
                    num_threads += 1;
                }
                self.optim_threads.store(cmp::min(self.max_threads, num_threads), atomic::Ordering::SeqCst);
            }
            if index + 1 >= self.optim_threads.load(atomic::Ordering::SeqCst) {
                self.is_blocked.store(true, atomic::Ordering::SeqCst);
            }
            index == 0
        } else {
            let index = self.inside_count.load(atomic::Ordering::SeqCst);
            let mut num_threads: usize = 1 + self.cvar.notify_all();
            num_threads = cmp::min(self.max_threads, num_threads);
            self.optim_threads.store(num_threads, atomic::Ordering::SeqCst);
            if index >= num_threads {
                self.is_blocked.store(true, atomic::Ordering::SeqCst);
            }
            true
        }
    }

    pub fn get_optim_threads(&self) -> usize {
        self.optim_threads.load(atomic::Ordering::SeqCst)
    }

    /// Release a resource from this semaphore.
    ///
    /// This will increment the number of resources in this semaphore by 1 and
    /// will notify any pending waiters in `acquire` or `access` if necessary.
    pub fn release(&self) {
        let mut count = self.lock.lock();

        *count -= 1;
        if self.inside_count.fetch_sub(1, atomic::Ordering::SeqCst) == 1 {
            self.optim_threads.store(self.max_threads, atomic::Ordering::SeqCst);
            self.is_blocked.store(false, atomic::Ordering::SeqCst);
            self.cvar.notify_one();
        }
    }

    /// Acquires a resource of this semaphore, returning an RAII guard to
    /// release the semaphore when dropped.
    ///
    /// This function is semantically equivalent to an `acquire` followed by a
    /// `release` when the guard returned is dropped.
    pub fn access(&self) -> SemarrierGuard {
        self.acquire();
        SemarrierGuard { sem: self }
    }
}

impl<'a> Drop for SemarrierGuard<'a> {
    fn drop(&mut self) {
        self.sem.release();
    }
}
