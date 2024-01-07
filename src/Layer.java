import java.util.ArrayList;

public class Layer {
    //Neurons in the layer
    public ArrayList<Neuron> neurons;
    public ArrayList<HiddenLayerNeuron> hiddenLayerNeurons;
    public ArrayList<OutputNeuron> outputLayerNeurons;
    Layer(ArrayList<Neuron> _neurons,int a){
        neurons = _neurons;
    }
    Layer(ArrayList<HiddenLayerNeuron> _hiddenLayerNeurons,double b){
        hiddenLayerNeurons = _hiddenLayerNeurons;
    }
    Layer(ArrayList<OutputNeuron> _outputLayerNeurons){
        outputLayerNeurons = _outputLayerNeurons;
    }
}