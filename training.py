import keras
from generator import DataGenerator
def basicTrain(model, input, output, epochs):
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    keras.backend.set_value(model.optimizer.learning_rate, 0.001)
    generator = DataGenerator(batch_size=64)
    model.fit(generator, epochs=epochs)


#callbackixx