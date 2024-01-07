public class HiddenLayerNeuron extends Neuron{
    public HiddenLayerNeuron(int _connectionCountToNextLayer,String weightsFilePath) {
        super(_connectionCountToNextLayer,weightsFilePath);
    }

    //Adjust the formula for different type of neural network structure. This formula is hard coded for a neural network consists of,
    //3 Input,
    //4 Hidden Layer,
    //1 Output Neuron
    public double CalculateHiddenLayerNeuronError(double weightedOutput, double outputLayerNeuronError, double hiddenLayerNeuronWeight){
        return weightedOutput * (1 - weightedOutput) * (outputLayerNeuronError * hiddenLayerNeuronWeight);
    }
}
