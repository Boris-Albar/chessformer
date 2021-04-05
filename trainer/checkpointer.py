import os
import pytorch_lightning as pl

class ChessformerCheckpointer(pl.Callback):

    def __init__(
        self,
        save_epoch_frequency,
        checkpoint_dir,
        prefix="Chessformer"
    ):

        self.save_epoch_frequency = save_epoch_frequency
        self.prefix = prefix
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, trainer: pl.Trainer, model):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if (epoch > 0) and (epoch % self.save_epoch_frequency == 0):
            filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            filename_onnx = filename.replace(".ckpt", ".onnx")
            ckpt_path = os.path.join(self.checkpoint_dir, filename)
            trainer.save_checkpoint(ckpt_path)
            model.save_to_onnx(os.path.join(self.checkpoint_dir, filename_onnx))
