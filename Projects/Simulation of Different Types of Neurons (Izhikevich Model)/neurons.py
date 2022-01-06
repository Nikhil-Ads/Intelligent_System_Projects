class AbstractNeuron:

    def __init__(self, V, tau, tspan):
        self.V = V
        self.tau = tau
        self.tspan = tspan

    def display_parameters(self):
        """Default implementation for displaying the parameters of a neuron, under Izhikevich model."""
        print(f"V = {self.V}, u = {self.b * self.V}, a = {self.a}, b = {self.b}, c = {self.c}, d = {self.d}")

    def simulate_neuron(self, ext_i):
        """This function represents the Izhikevich model for simulating any neuron. The parameters
        for the model are set before in the class and used as is for implementing the logic, as these parameters
        remain fixed for a particular type of neuron.

        :arg ext_i represents the external ext_i voltage given to the neuron for this simulation.
        :returns a tuple of functions: v(t), u(t) and spikes recorded.
        """
        # Initializing values of initial 'V' and 'u'
        V = self.V
        u = self.b * V

        vv = []
        uu = []
        spike_ts = []

        tau = self.tau
        for t in self.tspan:
            V = V + tau * ((0.04 * V ** 2) + (5 * V) + 140 - u + ext_i)
            u = u + tau * self.a * (self.b * V - u)

            if V >= 30:
                vv.append(30)
                V = self.c
                u = u + self.d
                spike_ts.append(1)
            else:
                vv.append(V)
                spike_ts.append(0)
            uu.append(u)
        return vv, uu, spike_ts


class RSNeuron(AbstractNeuron):
    # Defining parameter values for a 'Regular Spiking' neuron, as per the model
    a, b, c, d = 0.02, 0.2, -65, 8

    def display_parameters(self):
        print("Parameters for 'Regular Spiking' (RS) Neuron")
        super(RSNeuron, self).display_parameters()


class FSNeuron(AbstractNeuron):
    # Defining parameter values for a 'Fast Spiking' neuron, as per the model
    a, b, c, d = 0.1, 0.2, -65, 2

    def display_parameters(self):
        print("Parameters for 'Fast Spiking' (FS) Neuron")
        super(FSNeuron, self).display_parameters()


class CHNeuron(AbstractNeuron):
    # Defining parameter values for a 'Chattering' neuron, as per the model
    a, b, c, d = 0.02, 0.2, -50, 2

    def display_parameters(self):
        print("Parameters for 'Chattering' (CH) Neuron")
        super(CHNeuron, self).display_parameters()
