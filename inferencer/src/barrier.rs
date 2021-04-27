extern crate parking_lot_core;

use std::sync::{atomic, Arc};

struct BarrierInner {
    gsense: atomic::AtomicBool,
    count: atomic::AtomicUsize,
    max: atomic::AtomicUsize,
}

/// A barrier which enables multiple threads to synchronize the beginning of some computation.
pub struct Barrier {
    inner: Arc<BarrierInner>,
    lsense: bool,
    used: bool,
}

pub struct BarrierWaitResult(bool);

impl Barrier {
    pub fn new(n: usize) -> Self {
        Barrier {
            used: false,
            lsense: true,
            inner: Arc::new(BarrierInner {
                gsense: atomic::AtomicBool::new(true),
                count: atomic::AtomicUsize::new(0),
                max: atomic::AtomicUsize::new(n),
            }),
        }
    }

    pub fn wait(&mut self) -> BarrierWaitResult {
        self.used = true;
        self.lsense = !self.lsense;

        if self.inner.count.fetch_add(1, atomic::Ordering::SeqCst) == (self.inner.max.load(atomic::Ordering::SeqCst) - 1) {
            // we're the last to reach the barrier -- release all
            self.inner
                .count
                .store(0, atomic::Ordering::SeqCst);
            self.inner
                .gsense
                .store(self.lsense, atomic::Ordering::SeqCst);
            BarrierWaitResult(true)
        } else {
            // wait for everyone to reach the barrier
            let mut wait = parking_lot_core::SpinWait::new();
            while self.inner.gsense.load(atomic::Ordering::SeqCst) != self.lsense {
                // XXX: in theory we could go even further and park the thread eventually
                wait.spin();
            }
            BarrierWaitResult(false)
        }
    }

    pub fn set_max(&mut self, new_max: usize) {
        self.inner.max.store(new_max, atomic::Ordering::SeqCst);
    }
}

impl Clone for Barrier {
    fn clone(&self) -> Self {
        assert!(!self.used);
        Barrier {
            used: false,
            lsense: self.lsense,
            inner: self.inner.clone(),
        }
    }
}

impl BarrierWaitResult {
    /// Returns whether this thread from [`wait`] is the "leader thread".
    ///
    /// Only one thread will have `true` returned from their result, all other
    /// threads will have `false` returned.
    ///
    /// [`wait`]: struct.Barrier.html#method.wait
    ///
    /// # Examples
    ///
    /// ```
    /// use hurdles::Barrier;
    ///
    /// let mut barrier = Barrier::new(1);
    /// let barrier_wait_result = barrier.wait();
    /// assert_eq!(barrier_wait_result.is_leader(), true);
    /// ```
    pub fn is_leader(&self) -> bool {
        self.0
    }
}
