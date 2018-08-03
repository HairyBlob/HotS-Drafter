from graph_freezer import freezing_graph
from discriminator import train_discriminator
from estimator import train_estimator

discriminator = train_discriminator()
estimator = train_estimator()

freezing_graph("discriminator")
freezing_graph("estimator")