import keras
def basicTrain(model, input, output, epochs):
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    keras.backend.set_value(model.optimizer.learning_rate, 0.001)
    model.fit(x=input, y=output, batch_size=1, epochs=epochs)


#callbacki