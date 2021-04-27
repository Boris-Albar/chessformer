use chess::*;

/* for syzygy */
use shakmaty;
use shakmaty_syzygy::{Tablebase, Wdl, Dtz};

use ndarray::Array3;

use mcts::*;
use mcts::transposition_table::TranspositionHash;

use intmap::IntMap;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ChessGameStatus {
    Ongoing,
    WhiteWon,
    BlackWon,
    DrawByRepetion,
    Stalemate,
    DrawBy50Rule
}

#[derive(Clone, PartialEq)]
pub struct ChessGame {
    board: chess::Board,
    num_moves: u64,
    last_count_move: u64,
    repetition_number: u64,
    repetition_table: IntMap<u64>,
    syzygy_path: String,
}

impl ChessGame {
    pub fn new_game(syzygy_path: String) -> ChessGame {

        ChessGame {
            board: Board::default(),
            num_moves: 0,
            last_count_move: 0,
            repetition_number: 0,
            repetition_table: IntMap::with_capacity(512),
            syzygy_path: syzygy_path,
        }
    }

    pub fn is_game_drawned_by_50_rule(&self) -> bool {
        self.last_count_move == 100
    }

    pub fn is_game_drawned_by_repetition(&self) -> bool {
        self.repetition_number == 3
    }

    pub fn evaluate_material(&self) -> f64 {

        let game_status = self.game_status();

        if game_status == ChessGameStatus::Stalemate {
			0.0
		} else if game_status == ChessGameStatus::DrawBy50Rule {
            0.0
        } else if game_status == ChessGameStatus::DrawByRepetion {
            0.0
        } else if game_status == ChessGameStatus::WhiteWon {
			1.0
        } else if game_status == ChessGameStatus::BlackWon {
            -1.0
        } else {

            let bitboard_white = self.board.color_combined(Color::White);
            let bitboard_black = self.board.color_combined(Color::Black);

            let mut sum_white = 0.0;
            let mut sum_black = 0.0;

            sum_white += (self.board.pieces(Piece::Pawn) & bitboard_white).count() as f64;
            sum_white += ((self.board.pieces(Piece::Knight) & bitboard_white).count() as f64) * 3.0;
            sum_white += ((self.board.pieces(Piece::Bishop) & bitboard_white).count() as f64) * 3.0;
            sum_white += ((self.board.pieces(Piece::Rook) & bitboard_white).count() as f64) * 5.0;
            sum_white += ((self.board.pieces(Piece::Queen) & bitboard_white).count() as f64) * 9.0;

            sum_black += (self.board.pieces(Piece::Pawn) & bitboard_black).count() as f64;
            sum_black += ((self.board.pieces(Piece::Knight) & bitboard_black).count() as f64) * 3.0;
            sum_black += ((self.board.pieces(Piece::Bishop) & bitboard_black).count() as f64) * 3.0;
            sum_black += ((self.board.pieces(Piece::Rook) & bitboard_black).count() as f64) * 5.0;
            sum_black += ((self.board.pieces(Piece::Queen) & bitboard_black).count() as f64) * 9.0;

            (sum_white - sum_black) / 39.0
        }
    }

    pub fn game_status(&self) -> ChessGameStatus {
        let status = self.board.status();
        let side_to_move = self.board.side_to_move();

        let mut game_status = ChessGameStatus::Ongoing;

        if status == BoardStatus::Stalemate {
			game_status = ChessGameStatus::Stalemate;
        } else if self.is_game_drawned_by_50_rule() == true {
            game_status = ChessGameStatus::DrawBy50Rule;
		} else if self.is_game_drawned_by_repetition() == true {
            game_status = ChessGameStatus::DrawByRepetion;
		} else if status == BoardStatus::Checkmate {
			if side_to_move == Color::White {
				game_status = ChessGameStatus::BlackWon;
			} else {
				game_status = ChessGameStatus::WhiteWon;
			}
		}

        game_status
    }

    pub fn get_move_informations(&self, mov: ChessMove) -> (ChessGameStatus, bool, bool) {
        let is_capture = self.board.color_on(mov.get_dest()) == Some(!self.board.side_to_move());
        let is_pawn = self.board.piece_on(mov.get_source()) == Some(Piece::Pawn);
        let new_board = self.board.make_move_new(mov);
        let is_check = (self.board.checkers().popcnt() == 0) && (new_board.checkers().popcnt() > 0);

        let new_status = new_board.status();
        let side_to_move = self.board.side_to_move();

        let board_hash = new_board.get_hash();
        let board_rep_val = self.repetition_table.get(board_hash);

        let mut game_status = ChessGameStatus::Ongoing;

        if new_status == BoardStatus::Stalemate {
			game_status = ChessGameStatus::Stalemate;
        } else if (is_capture == false) && (is_pawn == false) && (self.last_count_move == 99)  {
            game_status = ChessGameStatus::DrawBy50Rule;
		} else if board_rep_val.is_some() && (*board_rep_val.unwrap() == 2) {
            game_status = ChessGameStatus::DrawByRepetion;
		} else if new_status == BoardStatus::Checkmate {
			if side_to_move == Color::White {
				game_status = ChessGameStatus::BlackWon;
			} else {
				game_status = ChessGameStatus::WhiteWon;
			}
		}

        (game_status, is_check, is_capture)
    }

    pub fn get_syzygy_wdl(&self) -> f64 {
        let mut eval = -5.0;

        if self.syzygy_path != "" {
            if self.board.combined().popcnt() <= 5 {
                let pos: shakmaty::Chess = self.board_string().parse::<shakmaty::fen::Fen>().expect("Cannot parse to shakmaty")
                .position(shakmaty::CastlingMode::Standard).expect("Cannot create shakmaty position!");

                /* Save the table object for reuse */
                let mut tables = Tablebase::new();
                tables.add_directory(&self.syzygy_path).expect("Cannot read syzygy tables!");

                let wdl = tables.probe_wdl(&pos).expect("Cannot probe position!");

                match wdl {
                    Wdl::Loss => eval = -1.0,
                    Wdl::BlessedLoss => eval = 0.0,
                    Wdl::Draw => eval = 0.0,
                    Wdl::CursedWin => eval = 0.0,
                    Wdl::Win => eval = 1.0,
                }
            }
        }

        eval
    }

    pub fn num_moves(&self) -> u64 {
        self.num_moves
    }

    pub fn fifty_rules_counter(&self) -> u64 {
        self.last_count_move
    }

    pub fn repetition_number(&self) -> u64 {
        self.repetition_number
    }

    pub fn board_string(&self) -> String {
        let fen: BoardBuilder = self.board.into();
        fen.to_string()
    }

    fn hash(&self) -> u64 {
        self.board.get_hash() ^ self.repetition_number ^ self.last_count_move
    }

    pub fn get_board_array(&self) -> Array3<f32> {
        let mut input_array = Array3::<f32>::zeros((1, 65, 15));

        for i in 0..63 {
            let square = unsafe { Square::new(i) };
            if let Some(piece) = self.board.piece_on(square) {
                if self.board.color_on(square).unwrap() == Color::White {
                    input_array[[0,i as usize, (piece.to_index() + 1) as usize]] = 1.0;
                } else {
                    input_array[[0,i as usize, (7 + piece.to_index() + 1) as usize]] = 1.0;
                }
            }
        }

        // side to move
        let side_to_move = self.board.side_to_move();
        if side_to_move == Color::White {
            input_array[[0,64,0]] = 1.0;
        } else {
            input_array[[0,64,0]] = 0.0;
        }

        let my_castle = self.board.my_castle_rights();
        let their_castle = self.board.their_castle_rights();
        // castling rights
        if side_to_move == Color::White { //white turn
            if my_castle == CastleRights::KingSide {
                input_array[[0,64,1]] = 1.0;
            } else if my_castle == CastleRights::QueenSide {
                input_array[[0,64,2]] = 1.0;
            } else if my_castle == CastleRights::Both {
                input_array[[0,64,1]] = 1.0;
                input_array[[0,64,2]] = 1.0;
            }

            if their_castle == CastleRights::KingSide {
                input_array[[0,64,3]] = 1.0;
            } else if my_castle == CastleRights::QueenSide {
                input_array[[0,64,4]] = 1.0;
            } else if my_castle == CastleRights::Both {
                input_array[[0,64,3]] = 1.0;
                input_array[[0,64,4]] = 1.0;
            }

        } else { // black turn
            if my_castle == CastleRights::KingSide {
                input_array[[0,64,3]] = 1.0;
            } else if my_castle == CastleRights::QueenSide {
                input_array[[0,64,4]] = 1.0;
            } else if my_castle == CastleRights::Both {
                input_array[[0,64,3]] = 1.0;
                input_array[[0,64,4]] = 1.0;
            }

            if their_castle == CastleRights::KingSide {
                input_array[[0,64,1]] = 1.0;
            } else if my_castle == CastleRights::QueenSide {
                input_array[[0,64,2]] = 1.0;
            } else if my_castle == CastleRights::Both {
                input_array[[0,64,1]] = 1.0;
                input_array[[0,64,2]] = 1.0;
            }
        }

        // repeat and fifty rule counter
        input_array[[0,64,5]] = (self.repetition_number as f32) / 3.0;
        input_array[[0,64,6]] = (self.last_count_move as f32) / 100.0;

        input_array
    }
}

impl TranspositionHash for ChessGame {
    fn hash(&self) -> u64 {
        self.hash()
    }
}

impl GameState for ChessGame {
    type Move = chess::ChessMove;
    type Player = chess::Color;
    type MoveList = Vec<chess::ChessMove>;

    fn current_player(&self) -> Self::Player {
        self.board.side_to_move()
    }

    fn available_moves(&self) -> Self::MoveList {
        MoveGen::new_legal(&self.board).collect()
    }

    fn make_move(&mut self, mov: &Self::Move) {

        let is_capture = self.board.color_on(mov.get_dest()) == Some(!self.board.side_to_move());
        let is_pawn = self.board.piece_on(mov.get_source()) == Some(Piece::Pawn);

        // TODO: this should be improved
		let board_cop = self.board.clone();
        let vec: Vec<chess::ChessMove> = MoveGen::new_legal(&board_cop).collect();
        board_cop.make_move(*mov, &mut self.board);

        self.num_moves += 1;

        // enable for debugging
        /*if !self.board.is_sane() && board_cop.is_sane() {
            let fen: BoardBuilder = board_cop.into();
            println!("{:?}", fen.to_string());
            println!("{:?}", self.board_string());
            println!("{:?}", board_cop.side_to_move());
            println!("{:?}", self.current_player());
            println!("{:?}", vec);
            println!("{:?}", *mov);
            println!("{:?}", vec.len());
            panic!("Invalid board!");
        }*/

        let board_hash = self.board.get_hash();
        if let Some(val) = self.repetition_table.get_mut(board_hash) {
            *val += 1;
            self.repetition_number = *val;
        } else {
            self.repetition_table.insert(board_hash, 1);
            self.repetition_number = 1;
        }

        if is_capture || is_pawn {
            self.last_count_move = 0;
        } else {
            self.last_count_move += 1;
        }

    }
}
