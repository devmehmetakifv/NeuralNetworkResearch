import java.util.ArrayList;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class NeuralNetwork {
    public Layer inputLayer;
    public Layer hiddenLayer;
    public Layer outputLayer;
    public ArrayList<Double> activatedHiddenLayerInputs;
    public double learnRate;

    public NeuralNetwork(Layer _inputLayer, Layer _hiddenLayer, Layer _outputLayer, double _learnRate) {
        activatedHiddenLayerInputs = new ArrayList<>();
        inputLayer = _inputLayer;
        hiddenLayer = _hiddenLayer;
        outputLayer = _outputLayer;
        learnRate = _learnRate;
    }

    public double ForwardPropagation(NeuralNetwork network,  int current_epoch) {
        double predictedOutcome;
        ArrayList<Integer> inputs = new ArrayList<>();
        for(int i = 0; i < network.inputLayer.neurons.size(); i++){
            inputs.add(network.inputLayer.neurons.get(i).inputVec.get(current_epoch));
        }

        ArrayList<Double> weightedInputs = new ArrayList<>();

        double _sum = 0;
        for(int k = 0; k < network.inputLayer.neurons.getFirst().getWeights().size(); k++){
            for(int j = 0; j < network.inputLayer.neurons.size(); j++){
                double weightedInput = inputs.get(j) * network.inputLayer.neurons.get(j).getWeight(k) /*+ network.inputLayer.neurons.get(j).getBias()*/;
                _sum += weightedInput;
            }
            weightedInputs.add(_sum);
            _sum=0;
        }

        for(int i = 0; i < weightedInputs.size(); i++){
            double activated = network.hiddenLayer.hiddenLayerNeurons.get(i).ReLUActivationFunction(weightedInputs.get(i));
            activatedHiddenLayerInputs.add(activated);
        }

        double lastWeightedInput = 0;
        for(int i = 0; i < network.hiddenLayer.hiddenLayerNeurons.getFirst().getWeights().size(); i++){
            double sum = 0;
            for(int j = 0; j < network.hiddenLayer.hiddenLayerNeurons.size(); j++){
                double weightedInput = activatedHiddenLayerInputs.get(j) * network.hiddenLayer.hiddenLayerNeurons.get(j).getWeight(i) /*+ network.hiddenLayer.hiddenLayerNeurons.get(j).getBias()*/;
                sum += weightedInput;
            }
            lastWeightedInput = sum;
        }
        predictedOutcome = network.outputLayer.outputLayerNeurons.getFirst().LinearActivationFunction(lastWeightedInput);

        return predictedOutcome;
    }
    public double Cost(double groundTruth, double predictedOutcome){
        return Math.pow((predictedOutcome - groundTruth),2);
    }
    public void updateWeightValuesTxt(ArrayList<Double> weights, String filePath){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            // Write each weight to the file
            for (double weight : weights) {
                writer.write(Double.toString(weight));
                writer.newLine();
            }
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
