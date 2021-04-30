use rand::prelude::*;
use rand_distr::Dirichlet;

use std::sync::{Arc};
use crate::fast_math::{sigmoid, tanh};

use mcts::*;
use mcts::tree_policy::*;
use mcts::transposition_table::*;

use chess;
use crate::chess_game::{ChessGame, ChessGameStatus};
use crate::trt_inferer::TrtInferer;

#[derive(Default)]
pub struct ChessformerTrtMCTS {
    v_loss: i64,
    visits_before_exp: u64,
}

impl ChessformerTrtMCTS {
    pub fn default(v_loss: i64, visits_before_expansion: u64) -> ChessformerTrtMCTS {
        ChessformerTrtMCTS {
            v_loss: v_loss,
            visits_before_exp: visits_before_expansion,
        }
    }
}

pub struct NNTrtEvaluator {
    inferer: Arc<TrtInferer>,
    root_player: chess::Color,
    temperature_policy: f64,
    dirichlet_epsilon: f64,
    dirichlet_alpha: f64,
}

impl NNTrtEvaluator {
    pub fn load_from_pointer(inferer: Arc<TrtInferer>, t_policy: f64, dirichlet_epsilon: f64, dirichlet_alpha: f64, root_player: chess::Color) -> NNTrtEvaluator {

        NNTrtEvaluator {
            inferer: inferer.clone(),
            root_player: root_player,
            temperature_policy: 1.0 / t_policy,
            dirichlet_epsilon: dirichlet_epsilon,
            dirichlet_alpha: dirichlet_alpha,
        }
    }
}

unsafe impl Sync for NNTrtEvaluator {}

impl Evaluator<ChessformerTrtMCTS> for NNTrtEvaluator {
    type StateEvaluation = f64;

    fn evaluate_new_state(&self, state: &ChessGame, moves: &Vec<chess::ChessMove>,
        _: Option<SearchHandle<ChessformerTrtMCTS>>, is_root: bool)
        -> (Vec<f64>, f64) {

        /* check game status and shortcut evaluation if needed */
        let status = state.game_status();
        if status == ChessGameStatus::WhiteWon {
            if self.root_player == chess::Color::White {
                return (Vec::<f64>::new(), 1.0);
            } else {
                return (Vec::<f64>::new(), -1.0);
            }
        } else if status == ChessGameStatus::BlackWon {
            if self.root_player == chess::Color::Black {
                return (Vec::<f64>::new(), 1.0);
            } else {
                return (Vec::<f64>::new(), -1.0);
            }
        } else if status != ChessGameStatus::Ongoing {
            return (Vec::<f64>::new(), 0.0);
        }

        let input_board = state.get_board_array();
        let output = self.inferer.get_batch_nn_evaluation(input_board, state.hash());
        //let output = self.get_nn_evaluation(input_board);

        let length = moves.len();
        let mut vec_moves = Vec::<f64>::with_capacity(length);

		for _leg_move in moves {

            let source = _leg_move.get_source().to_index() as i64;
            let dest = _leg_move.get_dest().to_index() as i64;
            let mut prob = 0.0;

            if let Some(piece) = _leg_move.get_promotion() {
                let is_white_promoting = (source < 20) as i64;

                match piece { // seventh rank
                    chess::Piece::Knight => prob = output[(4096 + (66 * is_white_promoting) + 0 + ((source % 8) * 3 - 1) + ((dest - source) - 8)) as usize],
                    chess::Piece::Bishop => prob = output[(4096 + (66 * is_white_promoting) + 22 + ((source % 8) * 3 - 1) + ((dest - source) - 8)) as usize],
                    chess::Piece::Rook => prob = output[(4096 + (66 * is_white_promoting) + 44 + ((source % 8) * 3 - 1) + ((dest - source) - 8)) as usize],
                    _ => prob = output[(source * 64 + dest) as usize],
                }
            } else {
                prob = output[(source * 64 + dest) as usize];
            }

			vec_moves.push((sigmoid(prob) as f64) / self.temperature_policy);
		}

        // normalize probabilities
        let sum_probs: f64 = vec_moves.iter().sum();
        vec_moves = vec_moves.iter().map(|x| x / sum_probs).collect();

        // if is root, add dirichlet noise
        if is_root == true && vec_moves.len() > 1 {
            let dirichlet = Dirichlet::new_with_size(self.dirichlet_alpha, vec_moves.len()).unwrap();
            let samples = dirichlet.sample(&mut rand::thread_rng());
            vec_moves = vec_moves.iter().enumerate().map(|(i, x)| (1.0 - self.dirichlet_epsilon) * x + self.dirichlet_epsilon * samples[i]).collect();
        }

        // get value from the neural network
        let mut state_eval = tanh(output[output.shape()[0] - 1]) as f64;

        // reverse policy for black
        if self.root_player == chess::Color::Black {
            state_eval = -1.0 * state_eval;
        }

        // if syzygy is enabled, correct the evaluation if possible
        let syzygy_wtz_value = state.get_syzygy_wdl();
        if syzygy_wtz_value != -5.0 {
            state_eval = syzygy_wtz_value;
        }

        (vec_moves, state_eval)
    }

    fn interpret_evaluation_for_player(&self, evaln: &f64, _player: &chess::Color) -> i64 {

		let adj_eval;

		if *_player == self.root_player {
			adj_eval = (*evaln * 1000.0) as i64
		} else {
			adj_eval = ((-1.0 * (*evaln)) * 1000.0) as i64
		}

		adj_eval
    }

    fn evaluate_existing_state(&self, _: &ChessGame,  evaln: &f64, _: SearchHandle<ChessformerTrtMCTS>) -> f64 {
        *evaln
    }
}

impl MCTS for ChessformerTrtMCTS {
    type State = ChessGame;
    type Eval = NNTrtEvaluator;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ApproxTable<Self>;

    fn virtual_loss(&self) -> i64 {
        self.v_loss
    }

    fn visits_before_expansion(&self) -> u64 {
        self.visits_before_exp
    }

    fn node_limit(&self) -> usize {
        std::usize::MAX
    }

    fn select_child_after_search<'a>(&self, children: &'a [MoveInfo<Self>]) -> &'a MoveInfo<Self> {
        children.into_iter().max_by_key(|child| child.visits()).unwrap()
    }

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}
