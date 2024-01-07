import java.util.ArrayList;

public class OutputNeuron extends Neuron{
    public ArrayList<Double> groundTruths;
    public OutputNeuron(ArrayList<Double> _groundTruths, int _connectionCountToNextLayer) {
        super(_connectionCountToNextLayer);
        groundTruths = new ArrayList<>();
        groundTruths = _groundTruths;
    }
    public double CalculateOutputNeuronError(double predictedOutput, double expectedOutput){
        return predictedOutput * (1 - predictedOutput) * (expectedOutput - predictedOutput);
    }
}
