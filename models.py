import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


def _base_model(alpha=1.0, weights='imagenet'):
	model = tfk.applications.mobilenet_v2.MobileNetV2(
	    include_top=False,
	    input_shape=(224, 224, 3),
	    alpha=alpha,
	    weights=weights,
	)
	return model


def warmup_model(momentum=0.9):
	base_model = _base_model()
	for layer in base_model.layers:
		if "BatchNormalization" in str(type(layer)):
			layer.trainable = True
			layer.momentum = momentum
		else:
			layer.trainable = False

	model = tfk.Sequential([
		base_model,
		tfkl.GlobalAveragePooling2D(),
		tfkl.Dense(1, activation='sigmoid'),
	])
	return model


def build_model(baseline_model, dropout=0.7):
	model = tfk.Sequential([
	    baseline_model,
	    #tfk.Model(inputs=mobilenet_model.inputs, outputs=mobilenet_output),
	    tfkl.GlobalAveragePooling2D(),
	    tfkl.Dropout(dropout),
	    tfkl.Dense(64, activation='relu'),
	    tfkl.Dropout(dropout),
	    tfkl.Dense(64, activation='relu'),
	    tfkl.Dropout(dropout),
	    tfkl.Dense(1, activation='sigmoid'),
	])
	return model
	