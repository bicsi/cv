import argparse
import loader
import models
import metrics
from loguru import logger as log
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


class ModelCheckpointCallback(tfk.callbacks.Callback):

	def __init__(self, period):
		super(ModelCheckpointCallback, self).__init__()

		self.period = period

	def on_epoch_end(self, epoch, logs=None):
		if (epoch + 1) % self.period != 0:
			return

		filename = f"weights/base/base-{epoch}.h5"
		log.info(f"Saving weights to {filename}...")

		base_model = self.model.layers[0]
		trainables = [layer.trainable for layer in base_model.layers]
		for layer in base_model.layers:
			layer.trainable = True
		base_model.save_weights(filename)
		for trainable, layer in zip(trainables, base_model.layers):
			layer.trainable = trainable


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--epochs', '-e', type=int, default=1000)
	parser.add_argument(
		'--steps-per-epoch', type=int, default=100)
	parser.add_argument(
		'--tiny', action='store_true')
	parser.add_argument(
		'--reset', action='store_true')
	parser.add_argument(
		'--checkpoint-period', type=int, default=10)
	return parser.parse_args()


def main(args):
	
	log.info("Loading training data...")
	load_data_kwargs = {}
	if args.tiny:
		load_data_kwargs["dim"] = "tiny"
	data = loader.load_train_data(**load_data_kwargs)
	train_gen, val_gen = loader.build_generators(data)
	log.success(f"{len(data['names'])} training examples successfully loaded.")

	log.info("Building model...")
	model = models.warmup_model(load_local=(not args.reset))
	print(model.summary())

	log.info("Compiling model...")
	model.compile(
    	optimizer=tfk.optimizers.Adam(learning_rate=1e-5), 
	    loss=metrics.f1_loss,
	    metrics=[metrics.precision, metrics.recall, metrics.f1],
	)

	log.info(f"Will save weights every {args.checkpoint_period} iterations")
	log.info("Training model...")
	model_checkpoint = ModelCheckpointCallback(args.checkpoint_period)
	model.fit_generator(
	    train_gen,
	    validation_data=val_gen,
	    steps_per_epoch=args.steps_per_epoch,
	    epochs=args.epochs,
	    callbacks=[model_checkpoint],
	)



if __name__ == "__main__":
	main(parse_args())