import matplotlib.pyplot as plt
from neurons import RSNeuron, FSNeuron, CHNeuron


class SimulateNeurons:
    """This class is the main class for the Assignment. It contains the starting point of the programs for the
    assignment. The class defines all the required methods and functions for plotting and doing the required tasks
    for the problems in the Assignment.

    For Problem 1:
        Method: simulate_rs_neuron() is defined.
    For Problem 2:
        Method: simulate_fs_neuron() is defined.
    For Problem 3:
        Method: simulate_ch_neuron() is defined.
    """

    # Defining number of steps to run
    steps = 1000
    step = 0.25 / steps

    # Defining initial voltage
    V = -64

    # Defining time step for sampling
    tau = 0.25
    tspan = [i * 0.25 for i in range(0, 4001)]

    T1 = 0

    def simulate_rs_neuron(self):
        """
        This method simulates an RS neuron, with the given voltage and time-step.
        :return: (voltage plot, R_VS_I plot) for RS neuron
        """
        # Initialize a 'Regular Spiking' (RS) neuron for simulating
        rs_neuron = RSNeuron(self.V, self.tau, self.tspan)
        # Defining the values of 'I' for which graphs have to be plotted.
        plots = [1, 10, 20, 30, 40]

        # Initializing data structures
        r_vs_i = {"R": [], "I": []}
        vv_plots = {}

        # Displaying parameters for 'RS' neuron
        rs_neuron.display_parameters()

        # Running Simulations for 'I' (External Inputs) from: 0 to 40
        for i in range(0, 41, 1):
            results = rs_neuron.simulate_neuron(i)
            r_vs_i["R"].append(sum(results[2][801:]) / 800)
            r_vs_i["I"].append(i)
            # Checking, if plots are required
            if i in plots:
                vv_plots[i] = results[0]
        self.plot_vv_v_t(vv_plots, "Simulation for 'Regular Spiking' (RS) Neuron")
        self.plot_r_v_i("'Spiking Rate' (R) VS 'External Input' (I) for 'RS' Neuron", RS=r_vs_i)
        return vv_plots, r_vs_i

    def simulate_fs_neuron(self, r_vs_i_rs_neuron):
        """
        This method simulates an FS neuron, with the given voltage and time-step.
        :return: (voltage plot, R_VS_I plot) for FS neuron
        """
        # Initialize a 'Fast Spiking' (FS) neuron for simulating
        neuron = FSNeuron(self.V, self.tau, self.tspan)
        # Defining the values of 'I' for which graphs have to be plotted.
        plots = [1, 10, 20, 30, 40]

        # Initializing data structures
        r_vs_i = {"R": [], "I": []}
        vv_plots = {}

        # Displaying parameters for 'FS' neuron
        neuron.display_parameters()

        # Running Simulations for 'I' (External Inputs) from: 0 to 40
        for i in range(0, 41, 1):
            results = neuron.simulate_neuron(i)
            r_vs_i["R"].append(sum(results[2][801:]) / 800)
            r_vs_i["I"].append(i)
            # Checking, if plots are required
            if i in plots:
                vv_plots[i] = results[0]
        self.plot_vv_v_t(vv_plots, "Simulation for 'Fast Spiking' (FS) Neuron")
        self.plot_r_v_i("'Spiking Rate' (R) VS 'External Input' (I): Comparison between 'RS' and 'FS' Neurons",
                        FS=r_vs_i, RS=r_vs_i_rs_neuron)
        return vv_plots, r_vs_i

    def simulate_ch_neuron(self):
        """
        This method simulates a CH neuron, with the given voltage and time-step.
        :return: (voltage plot, R_VS_I plot) for CH neuron
        """
        # Initialize a 'Chattering' (CH) neuron for simulating
        neuron = CHNeuron(self.V, self.tau, self.tspan)
        # Defining the values of 'I' for which graphs have to be plotted.
        plots = [1, 5, 10, 15, 20]

        # Initializing data structures
        vv_plots = {}

        # Displaying parameters for 'CH' neuron
        neuron.display_parameters()

        # Running Simulations for 'I' (External Inputs) for the given set of simulation values
        for i in plots:
            results = neuron.simulate_neuron(i)
            vv_plots[i] = results[0]
        self.plot_vv_v_t(vv_plots, "Simulation for 'Chattering' (CH) Neuron")
        return vv_plots

    def plot_vv_v_t(self, plots, title):
        """Plots graphs between 'VV' V/S 'T'. The plots passed are plotted in separate graphs as sub-plots in the
        same figure. The entire figure is given labels which are common for all the graphs. As we are plotting 'VV' (
        Voltage in neuron) against 'T' (Time steps), the labels for the axes are fixed.

        The title of the figure is passed which is used to provide the title of the figure.
        """
        fig, subplots = plt.subplots(len(plots), sharex='col')
        fig.suptitle(title)
        fig.tight_layout()
        fig.supxlabel("Time Steps")
        fig.supylabel("Voltage in the neuron")
        for ((i, VV), j) in zip(plots.items(), range(len(plots))):
            subplots[j].set_title(f"Plot of 'VV', when External Input: {i}")
            subplots[j].set_ylim(-75, 40)
            subplots[j].plot(self.tspan, VV)
        plt.show()

    def plot_r_v_i(self, title, **plots):
        """PLots R V/S I plots"""
        for label, df in plots.items():
            plt.plot(df['I'], df['R'], label=label)
        plt.legend(loc="upper left")
        plt.title(title)
        plt.xlabel("External Input (I)")
        plt.ylabel("Spiking Rate (R)")
        plt.show()


if __name__ == '__main__':
    # Initializing Simulator for simulating different neurons.
    simulate_neurons = SimulateNeurons()
    rs_neuron_outputs = simulate_neurons.simulate_rs_neuron()
    fs_neuron_outputs = simulate_neurons.simulate_fs_neuron(rs_neuron_outputs[1])
    ch_neuron_outputs = simulate_neurons.simulate_ch_neuron()
