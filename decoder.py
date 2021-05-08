from tensorflow.keras.layers import UpSampling2D, Reshape, Conv2D, Input, Dense
from tensorflow.keras import Model

def Decoder(z_dim):
    inputs  = Input(shape=[z_dim])
    x = inputs    
    x = Dense(7*7*64, activation='relu')(x)
    x = Reshape((7,7,64))(x)

    x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D((2,2))(x)
    
    x = Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D((2,2))(x)    

    out = Conv2D(filters=1, kernel_size=(3,3), strides=1, padding='same', activation='sigmoid')(x)
       
    return Model(inputs=inputs, outputs=out, name='decoder')