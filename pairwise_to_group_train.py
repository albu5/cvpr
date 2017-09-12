from keras.layers import Dense, Input, Reshape, Lambda
from keras.models import Model
from keras.backend import square
from scipy.io import loadmat
from keras import losses

# testing loadmat
# int01 = loadmat('pairwise_int/int01.mat')
# print(int01)
# interaction = int01['interaction']
# print(interaction.shape)

max_people = 20

"""
========================================================================================================================
LOSSES DEFINITION
========================================================================================================================
"""


def frobenius_loss(y_true, y_pred):
    squared_error = square(y_true-y_pred[0])
    # y_pred is predicted groups and y_pred[1] is masking_matrix
    return squared_error*y_pred[1]


losses.frobenius_loss = frobenius_loss

"""
========================================================================================================================
NETWORK DEFINITION
========================================================================================================================
"""
masking_matrix = Input(shape=(max_people, max_people), name='input_mask')
masking_matrix_out = Reshape(target_shape=(max_people, max_people), name='random_layer')(masking_matrix)
# since this has to be used as output too

pairwise_interaction = Input(shape=(max_people, max_people), name='pairwise_interaction_input')
pairwise_interaction_flat = Reshape(target_shape=(max_people*max_people,),
                                    name='pairwise_interaction_input_flat')(pairwise_interaction)

# can add one more dense layer here
group_belonging_flat = Dense(max_people*max_people,
                             activation='sigmoid',
                             name='group_belonging_flat')(pairwise_interaction_flat)
group_belonging = Reshape(target_shape=(max_people, max_people),
                          name='group_belonging')(group_belonging_flat)

pairwise_to_group_net = Model(inputs=[pairwise_interaction, masking_matrix],
                              outputs=[group_belonging, masking_matrix])

print(pairwise_to_group_net.summary())
pairwise_to_group_net.compile(optimizer='adam', loss=losses.frobenius_loss)
