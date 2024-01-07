import com.sun.source.tree.ArrayAccessTree;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        //Let's read from our dataset and extract x1,x2 and x3 values.
        String datasetPath = "/home/xenon/IdeaProjects/NeuralNetwork/src/dataset.csv";
        String inputNeuron1_weightValuesTxt = "/home/xenon/IdeaProjects/NeuralNetwork/WeightValues/InputNeuron1_weightValues.txt";
        String inputNeuron2_weightValuesTxt = "/home/xenon/IdeaProjects/NeuralNetwork/WeightValues/InputNeuron2_weightValues.txt";
        String inputNeuron3_weightValuesTxt = "/home/xenon/IdeaProjects/NeuralNetwork/WeightValues/InputNeuron3_weightValues.txt";

        String HiddenNeuron1_weightValuesTxt = "/home/xenon/IdeaProjects/NeuralNetwork/WeightValues/HiddenNeuron1_weightValues.txt";
        String HiddenNeuron2_weightValuesTxt = "/home/xenon/IdeaProjects/NeuralNetwork/WeightValues/HiddenNeuron2_weightValues.txt";
        String HiddenNeuron3_weightValuesTxt = "/home/xenon/IdeaProjects/NeuralNetwork/WeightValues/HiddenNeuron3_weightValues.txt";
        String HiddenNeuron4_weightValuesTxt = "/home/xenon/IdeaProjects/NeuralNetwork/WeightValues/HiddenNeuron4_weightValues.txt";

        ArrayList<String> inputNeuronsWeightTxts = new ArrayList<>();
        inputNeuronsWeightTxts.add(inputNeuron1_weightValuesTxt);
        inputNeuronsWeightTxts.add(inputNeuron2_weightValuesTxt);
        inputNeuronsWeightTxts.add(inputNeuron3_weightValuesTxt);

        ArrayList<String> hiddenNeuronsWeightTxts = new ArrayList<>();
        hiddenNeuronsWeightTxts.add(HiddenNeuron1_weightValuesTxt);
        hiddenNeuronsWeightTxts.add(HiddenNeuron2_weightValuesTxt);
        hiddenNeuronsWeightTxts.add(HiddenNeuron3_weightValuesTxt);
        hiddenNeuronsWeightTxts.add(HiddenNeuron4_weightValuesTxt);

        //Start by generating neuron lists for each layer.
        ArrayList<Neuron> inputNeurons = new ArrayList<>();
        ArrayList<HiddenLayerNeuron> hiddenLayerNeurons = new ArrayList<>();
        ArrayList<OutputNeuron> outputLayerNeurons = new ArrayList<>();

        ArrayList<String> x1Values = new ArrayList<>();
        ArrayList<String> x2Values = new ArrayList<>();
        ArrayList<String> x3Values = new ArrayList<>();
        ArrayList<String> groundTruths_String = new ArrayList<>();

        extractDataset(x1Values, x2Values, x3Values, groundTruths_String, datasetPath);

        //Let's populate each neuron list for the desired network structure. For this problem,
        // -> 3 input neurons
        // -> 10 hidden layer neurons
        // -> 1 output layer neuron
        Neuron inpNeuron_0 = new Neuron(convertToIntegerList(x1Values),4,inputNeuron1_weightValuesTxt);
        Neuron inpNeuron_1 = new Neuron(convertToIntegerList(x2Values),4,inputNeuron2_weightValuesTxt);
        Neuron inpNeuron_2 = new Neuron(convertToIntegerList(x3Values),4,inputNeuron3_weightValuesTxt);
        inputNeurons.add(inpNeuron_0);
        inputNeurons.add(inpNeuron_1);
        inputNeurons.add(inpNeuron_2);

        for(int i = 0; i < 4; i++) {
            HiddenLayerNeuron neuron = new HiddenLayerNeuron(1,hiddenNeuronsWeightTxts.get(i));
            hiddenLayerNeurons.add(neuron);
        }

        OutputNeuron outputNeuron = new OutputNeuron(convertToDoubleList(groundTruths_String),1);
        outputLayerNeurons.add(outputNeuron);

        //Now, create layers and populate layers with neurons.
        InputLayer inputLayer = new InputLayer(inputNeurons,1);
        HiddenLayer hiddenLayer = new HiddenLayer(hiddenLayerNeurons,1);
        OutputLayer outputLayer = new OutputLayer(outputLayerNeurons);

        //Let's go ahead and initialize the neural network.
        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, hiddenLayer, outputLayer, 0.005);
        ArrayList<Double> groundTruths = neuralNetwork.outputLayer.outputLayerNeurons.getFirst().groundTruths;
        ArrayList<Double> predictedOutcomes = new ArrayList<>();
        ArrayList<Double> calculatedErrors = new ArrayList<>();
        ArrayList<int[]> inputArrays = new ArrayList<>();
        ArrayList<double[]> activatedHiddenLayerInputArrays = new ArrayList<>();
        ArrayList<Double> deltaWeights = new ArrayList<>();

        int total_epoch = 5;

        for(int current_epoch = 0; current_epoch < total_epoch; current_epoch++) {
            double predictedOutcome = neuralNetwork.ForwardPropagation(neuralNetwork, current_epoch);
            predictedOutcomes.add(predictedOutcome);

            //Get inputs for each epoch with arrays and store all arrays in an ArrayList

            int[] inputs = new int[neuralNetwork.inputLayer.neurons.size()];
            double[] activatedInputs = new double[neuralNetwork.activatedHiddenLayerInputs.size()];
            for(int i = 0; i < neuralNetwork.inputLayer.neurons.size(); i++){
                inputs[i] = neuralNetwork.inputLayer.neurons.get(i).inputVec.get(current_epoch);
            }
            for(int i = 0; i < neuralNetwork.activatedHiddenLayerInputs.size(); i++){
                activatedInputs[i] = neuralNetwork.activatedHiddenLayerInputs.get(i);
            }
            inputArrays.add(inputs);
            activatedHiddenLayerInputArrays.add(activatedInputs);
        }

        for(int i = 0; i < total_epoch; i++){
            calculatedErrors.add(groundTruths.get(i) - predictedOutcomes.get(i));
        }
        for (int j = 0; j < neuralNetwork.inputLayer.neurons.size() + neuralNetwork.hiddenLayer.hiddenLayerNeurons.size(); j++) {
            double sums = 0;
            for (int i = 0; i < calculatedErrors.size(); i++) {
                if (j >= neuralNetwork.inputLayer.neurons.size()) {
                    sums += calculatedErrors.get(i) * activatedHiddenLayerInputArrays.get(i)[j - neuralNetwork.inputLayer.neurons.size()];
                } else {
                    sums += calculatedErrors.get(i) * inputArrays.get(i)[j];
                }
            }
            deltaWeights.add(sums);
        }
        for(int i = 0; i < deltaWeights.size(); i++){
            deltaWeights.set(i,deltaWeights.get(i) * neuralNetwork.learnRate);
        }

        for(int i = 0; i < neuralNetwork.inputLayer.neurons.size(); i++){
            for(int j = 0; j < neuralNetwork.inputLayer.neurons.getFirst().getWeights().size(); j++){
                neuralNetwork.inputLayer.neurons.get(i).setWeight(deltaWeights.get(i) , j);
            }
        }
        for(int i = 0; i < neuralNetwork.hiddenLayer.hiddenLayerNeurons.size(); i++){
            for(int j = 0; j < neuralNetwork.hiddenLayer.hiddenLayerNeurons.getFirst().getWeights().size(); j++){
                neuralNetwork.hiddenLayer.hiddenLayerNeurons.get(i).setWeight(deltaWeights.get(i-1+neuralNetwork.inputLayer.neurons.size()) , j);
            }
        }
        //Update weight values
        for(int i = 0; i < neuralNetwork.inputLayer.neurons.size(); i++){
            neuralNetwork.updateWeightValuesTxt(neuralNetwork.inputLayer.neurons.get(i).getWeights(),inputNeuronsWeightTxts.get(i));
        }
        for(int i = 0; i < neuralNetwork.hiddenLayer.hiddenLayerNeurons.size(); i++){
            neuralNetwork.updateWeightValuesTxt(neuralNetwork.hiddenLayer.hiddenLayerNeurons.get(i).getWeights(), hiddenNeuronsWeightTxts.get(i));
        }
        for(int i = 0; i < predictedOutcomes.size(); i++){
            System.out.println("AI predicted: " + predictedOutcomes.get(i) + " True answer: " + groundTruths.get(i));
        }


    }

    private static ArrayList<Double> convertToDoubleList(ArrayList<String> stringList) {
        ArrayList<Double> doubleList = new ArrayList<>();
        for (String value : stringList) {
            try {
                double parsedValue = Double.parseDouble(value);
                doubleList.add(parsedValue);
            } catch (NumberFormatException e) {
                // Handle parsing exceptions if needed
                e.printStackTrace();
            }
        }
        return doubleList;
    }
    private static ArrayList<Integer> convertToIntegerList(ArrayList<String> stringList) {
        ArrayList<Integer> integerList = new ArrayList<>();
        for (String value : stringList) {
            try {
                int parsedValue = Integer.parseInt(value);
                integerList.add(parsedValue);
            } catch (NumberFormatException e) {
                // Handle parsing exceptions if needed
                e.printStackTrace();
            }
        }
        return integerList;
    }

    private static void extractDataset(ArrayList<String> x1Values,ArrayList<String> x2Values,ArrayList<String> x3Values, ArrayList<String> groundTruths, String datasetPath) {
        try (BufferedReader br = new BufferedReader(new FileReader(datasetPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] columns = line.split(",");

                x1Values.add(columns[0].trim());
                x2Values.add(columns[1].trim());
                x3Values.add(columns[2].trim());

                groundTruths.add(columns[3].trim());

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void Debugging(NeuralNetwork neuralNetwork,ArrayList<Neuron> inputNeurons, ArrayList<HiddenLayerNeuron> hiddenLayerNeurons, ArrayList<OutputNeuron> outputLayerNeurons){
        //Debugging to check if forward propagation algorithm is true.
        System.out.println("---------------------------------------------------------------------------------------------");
        System.out.println("1st prediction values:");
        System.out.println("1st input: " + neuralNetwork.inputLayer.neurons.get(0).inputVec.get(0));
        System.out.println("2nd input: " + neuralNetwork.inputLayer.neurons.get(1).inputVec.get(0));
        System.out.println("3rd input: " + neuralNetwork.inputLayer.neurons.get(2).inputVec.get(0));
        System.out.println("---------------------------------------------------------------------------------------------");
        System.out.println("2nd prediction values:");
        System.out.println("1st input: " + neuralNetwork.inputLayer.neurons.get(0).inputVec.get(1));
        System.out.println("2nd input: " + neuralNetwork.inputLayer.neurons.get(1).inputVec.get(1));
        System.out.println("3rd input: " + neuralNetwork.inputLayer.neurons.get(2).inputVec.get(1));
        System.out.println("---------------------------------------------------------------------------------------------");
        System.out.println("3rd prediction values:");
        System.out.println("1st input: " + neuralNetwork.inputLayer.neurons.get(0).inputVec.get(2));
        System.out.println("2nd input: " + neuralNetwork.inputLayer.neurons.get(1).inputVec.get(2));
        System.out.println("3rd input: " + neuralNetwork.inputLayer.neurons.get(2).inputVec.get(2));
        System.out.println("---------------------------------------------------------------------------------------------");
        for(int i = 0; i < inputNeurons.size(); i++){
            System.out.println("Input Neuron #"+i+": ");
            //System.out.println("Bias: " + inputNeurons.get(i).getBias());
            for(int j = 0; j < inputNeurons.get(i).getWeights().size(); j++){
                System.out.println("Weight #" + j + ": " + inputNeurons.get(i).getWeight(j));
            }
            System.out.println();
        }

        System.out.println("---------------------------------------------------------------------------------------------");

        for(int i = 0; i < hiddenLayerNeurons.size(); i++){
            System.out.println("Hidden Layer Neuron #"+i+": ");
            //System.out.println("Bias: " + hiddenLayerNeurons.get(i).getBias());
            for(int j = 0; j < hiddenLayerNeurons.get(i).getWeights().size(); j++){
                System.out.println("Weight #" + j + ": " + hiddenLayerNeurons.get(i).getWeight(j));
            }
            System.out.println();
        }

        System.out.println("---------------------------------------------------------------------------------------------");

        for(int i = 0; i < outputLayerNeurons.size(); i++){
            System.out.println("Output Layer Neuron #"+i+": ");
            //System.out.println("Bias: " + outputLayerNeurons.get(i).getBias());
            for(int j = 0; j < outputLayerNeurons.get(i).getWeights().size(); j++){
                System.out.println("Weight #" + j + ": " + outputLayerNeurons.get(i).getWeight(j));
            }
            System.out.println();
        }
    }

}
