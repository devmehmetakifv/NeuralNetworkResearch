import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.Random;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Neuron {
    private final double LEAKY_RELU_CONSTANT = 0.01;
    private ArrayList<Double> weights;
    //private double bias;
    private int connectionCountToNextLayer;
    public ArrayList<Integer> inputVec;
    Random random;

    //Constructor for input neurons.
    public Neuron(ArrayList<Integer> _inputVec, int _connectionCountToNextLayer, String weightsFilePath){
        random = new Random();
        weights = new ArrayList<>();
        inputVec = _inputVec; // Assigning the passed _inputVec parameter
        connectionCountToNextLayer = _connectionCountToNextLayer; // Assign the passes _connectionCountToNextLayer parameter
        int count = readWeightsFromFile(weightsFilePath);
        if(count <= 0){
            // Initialize weight and bias values randomly for the initial run
            if(connectionCountToNextLayer == 1){
                weights.add(random.nextDouble() * 2 - 1);
            }
            else{
                for(int i = 0; i < connectionCountToNextLayer; i++) { // Use the passed parameter here
                    weights.add(random.nextDouble() * 2 - 1);
                }
            }
        }
        /*
        bias = random.nextDouble() * 0.02 - 0.01;
        */
    }


    //Constructor for hidden layer neurons.
    public Neuron(int _connectionCountToNextLayer, String weightsFilePath) {
        random = new Random();
        weights = new ArrayList<>();
        inputVec = new ArrayList<>();
        connectionCountToNextLayer = _connectionCountToNextLayer;
        int count = readWeightsFromFile(weightsFilePath);
        if(count <= 0){
            // Initialize weight and bias values randomly for the initial run
            if(connectionCountToNextLayer == 1){
                weights.add(random.nextDouble() * 2 - 1);
            }
            else{
                for(int i = 0; i < connectionCountToNextLayer; i++) { // Use the passed parameter here
                    weights.add(random.nextDouble() * 2 - 1);
                }
            }
        }
        /*
        bias = random.nextDouble() * 0.02 - 0.01;
         */

    }

    public Neuron(int _connectionCountToNextLayer) {
        connectionCountToNextLayer = _connectionCountToNextLayer;
    }


    /*public void setBias(double biasVal) {
        bias = biasVal;
    }
     */

    public double getWeight(int weightIndex) {
        return weights.get(weightIndex);
    }

    public ArrayList<Double> getWeights(){
        return weights;
    }

    public void setWeight(double weight, int weightIndex){
        weights.set(weightIndex,weight);
    }
    public void setWeights(ArrayList<Double> _weights){weights = _weights;}

    /*public double getBias() {
        return bias;
    }

     */

    /*
    public double calculateFinalValue(ArrayList<Double> inputs) {
        double sum = 0;
        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs.get(i) * getWeight(i);
        }
        return sum + getBias();
    } */

    public double ReLUActivationFunction(double weightedSum) {
        if(weightedSum > 0){
            return weightedSum;
        }
        else{
            return weightedSum * LEAKY_RELU_CONSTANT;
        }
    }

    public double LinearActivationFunction(double weightedSum) {
        return weightedSum; // Linear activation (identity function)
    }

    public int readWeightsFromFile(String fileName) {
        ArrayList<Double> weightList = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // Convert the read line to a Double and add it to the ArrayList
                double weight = Double.parseDouble(line);
                weightList.add(weight);
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }

        // Set the weights directly as ArrayList
        setWeights(weightList);
        return weightList.size();
    }
}
