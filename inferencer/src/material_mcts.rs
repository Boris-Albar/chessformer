use mcts::*;
use mcts::tree_policy::*;
use mcts::transposition_table::*;

use chess;
use crate::chess_game::ChessGame;

#[derive(Default)]
pub struct ChessMaterialMCTS;

pub struct MaterialEvaluator;

impl Evaluator<ChessMaterialMCTS> for MaterialEvaluator {
    type StateEvaluation = f64;

    fn evaluate_new_state(&self, state: &ChessGame, moves: &Vec<chess::ChessMove>,
        _: Option<SearchHandle<ChessMaterialMCTS>>, is_root: bool)
        -> (Vec<f64>, f64) {

        //let state_eval: f64 = state.evaluate_material();
        let length = moves.len() as f64;
        let mut vec_moves = Vec::with_capacity(length as usize);

		for _leg_move in moves {
			vec_moves.push(1.0 / length);
		}

        (vec_moves, 0.0)
    }

    fn interpret_evaluation_for_player(&self, evaln: &f64, _player: &chess::Color) -> i64 {
		let adj_eval;

		if *_player == chess::Color::White {
			adj_eval = (*evaln * 1000.0) as i64
		} else {
			adj_eval = ((1.0 - *evaln) * 1000.0) as i64
		}

		adj_eval
    }

    fn evaluate_existing_state(&self, _: &ChessGame,  evaln: &f64, _: SearchHandle<ChessMaterialMCTS>) -> f64 {
        *evaln
    }
}

impl MCTS for ChessMaterialMCTS {
    type State = ChessGame;
    type Eval = MaterialEvaluator;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ApproxTable<Self>;

    fn virtual_loss(&self) -> i64 {
        500
    }

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}


