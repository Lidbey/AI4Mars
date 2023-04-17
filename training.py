import keras
from generator import DataGenerator
def basicTrain(model, input, output, epochs):
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    keras.backend.set_value(model.optimizer.learning_rate, 0.001)
    generator = DataGenerator(n=6, batch_size=1)
    model.fit(generator, epochs=epochs)


#callbacki