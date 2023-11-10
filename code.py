def CNN_GRU_Model(x_train, Y, x_test, Y_test, steps, features, outputs):
    '''
    CNN model with attention mechanism and GRU
    '''
    print('Running the CNN-GRU model with attention and GRU...')

    # Input layer
    input_layer = Input(shape=(steps, features))

    # Convolutional layers
    conv1 = Conv1D(32, kernel_size=1, activation='relu')(input_layer)
    conv2 = Conv1D(128, kernel_size=1, activation='relu')(conv1)

    # Attention mechanism
    query = Conv1D(32, kernel_size=1, activation='relu')(conv2)
    key = Conv1D(32, kernel_size=1, activation='relu')(conv2)
    value = Conv1D(128, kernel_size=1, activation='relu')(conv2)

    attention_scores = Multiply()([query, key])
    attention_weights = Softmax(axis=1)(attention_scores)

    # Reshape attention weights for element-wise multiplication
    attention_weights = Flatten()(attention_weights)
    attention_weights = Dense(steps * features)(attention_weights)
    # Adjust the shape of attention_weights_adjusted
    attention_weights_adjusted = Reshape((steps, features, 1))(attention_weights)
    print("Attention weights shape:", attention_weights_adjusted.shape)

    # Element-wise multiplication
    attention_mul = Multiply()([attention_weights_adjusted, value])

    # Reshape attention_mul to match the GRU input shape
    gru_input = Reshape((steps, features*128))(attention_mul)

    # GRU layer
    gru = GRU(64, return_sequences=True)(gru_input)
    
    # Flatten and fully connected layers
    flatten = Flatten()(gru)
    dense1 = Dense(32, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    dense3 = Dense(128, activation='relu')(dense2)

    # Output layer
    output_layer = Dense(outputs)(dense3)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(lr=0.001)
    model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    print(model.summary())
    
    # Plot the model architecture
    # plot_model(model, to_file='E:/CXC/Rapid_FloodModelling_CNN-master/Data/Graph.png', dpi=1200)
    
    # Start time
    start = timeit.default_timer()
    
    # Early stopping callback
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    
    # Train the model
    history = model.fit(x_train, Y, validation_data=(x_test, Y_test), batch_size=10, callbacks=[monitor], verbose=0, epochs=100)
    
    # Stop time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    
    # Plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    return model

export_pred_data(locations)