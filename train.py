import argparse
import loader
import models
import metrics
from loguru import logger as log
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


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

	model = None
	if not args.reset:
		try:
			log.info("Loading model...")
			model = tfk.models.load_model(
			    "weights/main",
			    compile=False)
		except Exception as ex:
			log.error("Error loading model from disk.")
			log.error(ex)
			model = None

	if model is None:
		log.info("Building model...")
		model = models.main_model(load_local=(not args.reset))
		print(model.summary())

	log.info("Compiling model...")
	model.compile(
    	optimizer=tfk.optimizers.Adam(learning_rate=1e-5), 
	    loss=metrics.f1_loss,
	    metrics=[metrics.precision, metrics.recall, metrics.f1],
	)

	log.info(f"Will save weights every {args.checkpoint_period} iterations")
	log.info("Training model...")
	model_checkpoint = tfk.callbacks.ModelCheckpoint(
		"weights/main", 
		monitor='val_loss', 
		verbose=1, 
		save_best_only=True, 
		period=args.checkpoint_period,
	)
	model.fit_generator(
	    train_gen,
	    validation_data=val_gen,
	    steps_per_epoch=args.steps_per_epoch,
	    epochs=args.epochs,
	    callbacks=[model_checkpoint],
	)



if __name__ == "__main__":
	main(parse_args())