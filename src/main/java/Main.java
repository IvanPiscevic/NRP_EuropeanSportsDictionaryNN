/*
# europeanSportsDictionaryNN - A java neural network program used for translation of sport related words from
#                              different european languages to croatian language.
#                              Uses Encog library to build neural networks.
#                              Created by Tomislav Horvat and Ivan Piščević, TVZ.
# For every croatian word there is foreign 8 translations:
# english, german, french, italian, spanish, portuguese,
# dutch, czech, polish, danish, swedish, norwegian
*/

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;

import java.io.*;
import java.util.*;

public class Main {

    public static final int NUMBER_OF_FOREIGN_PER_CRO_WORDS = 8;   // 8 for slavic or 12 for foreign
    public static final int NUMBER_OF_WORD_ATTRIBUTES = 16;

    public static int input_neurons = NUMBER_OF_WORD_ATTRIBUTES;
    public static int hidden_neurons = 80;
    public static int hidden_neurons_2nd = 80;
    public static int output_neurons = 12;

    public static void main(String[] args) {
        // Initialization
        List<String> croatianWordsList = readWordsFile("croSportWords.txt");
        List<String> foreignWordsList = readWordsFile("slavicWordsTrain.txt");
        List<String> foreignWordsTest = readWordsFile("slavicWordsTest.txt");

        int numCroWords = croatianWordsList.size();
        int numForeignWords = foreignWordsList.size();
        int numForeignWordsTest = foreignWordsTest.size();

        double[][] input_matrix = new double[numForeignWords][NUMBER_OF_WORD_ATTRIBUTES];
        double[][] output_matrix = new double[numForeignWords][numCroWords];
        double[][] input_matrix_test = new double[numForeignWordsTest][NUMBER_OF_WORD_ATTRIBUTES];

        calculateOneHotOutput(output_matrix);

        // Calculate attribute values
        calculateWordAttributes(input_matrix, foreignWordsList);
        calculateWordAttributes(input_matrix_test, foreignWordsTest);

        // Min-Max normalization of input data
        normalize(input_matrix);
        normalize(input_matrix_test);

        // NN Initialization
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, input_neurons));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hidden_neurons));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hidden_neurons_2nd));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, output_neurons));
        network.getStructure().finalizeStructure();
        network.reset();

        // Training Dataset Initialization
        BasicMLDataSet trainingDataSet = new BasicMLDataSet(input_matrix, output_matrix);

        // Training NN
        final Backpropagation train = new Backpropagation(network, trainingDataSet);


        int epoch = 1;

        do {
            train.iteration();
            double error_perc = train.getError() * 100;
            System.out.println("Epoch #" + epoch + " Error:" + error_perc + "%");
            epoch++;
        } while(train.getError() > 0.01);
        train.finishTraining();

        System.out.println("=== NN Results ===");
        System.out.println("Foreign word = Croatian translation");
        double results = 0;
        List<Double> result_list = new ArrayList<>();
        for (double[] words : input_matrix_test) {
            double[] outputTest = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            network.compute(words, outputTest);
//            results += calculateNNResults(outputTest, croatianWordsList, foreignWordsTest);
            result_list.add(calculateNNResults(outputTest, croatianWordsList, foreignWordsTest));
        }

        System.out.println(result_list);
//        System.out.println("Rezultati testnog seta: " + ((results/input_matrix_test.length) * 100));

        for (int i = 0; i < result_list.size(); i++) {
            if (i == result_list.get(i)) {
                results += 1;
            }
        }

        System.out.println("Rezultati testnog seta: " + ((results/input_matrix_test.length) * 100));

        Encog.getInstance().shutdown();
    }

    private static List<String> readWordsFile(String path) {
        String line = "";
        List<String> foreignWordsList = new LinkedList<>();
        File foreignWordsInputFile = new File(path);

        try {
            BufferedReader in = new BufferedReader(new FileReader(foreignWordsInputFile));
            while ((line = in.readLine()) != null) {
                if (!line.contains("=")) {
                    foreignWordsList.add(line);
                }
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return foreignWordsList;
    }
    private static void calculateOneHotOutput(double[][] output_matrix) {

        int x = output_matrix[0].length;
        int y = output_matrix.length;
        int counter = 0, x_pos = 0;   // Every 12 elements move to the right

        System.out.println(output_matrix.length);
        System.out.println(output_matrix[0].length);

        // Initializing all of the output matrix fields to 0
        for (int i = 0; i < y; i++) {
            for (int j = 0; j < x; j++) {
                output_matrix[i][j] = 0;
            }
        }

        // One-hot encoding every foreign word to appropriate croatian word
        for (int i = 0; i < y; i++) {
            if (counter > NUMBER_OF_FOREIGN_PER_CRO_WORDS - 1) {
                counter = 0;
                x_pos++;
            }
            output_matrix[i][x_pos] = 1;
            counter++;
        }
    }
    private static void calculateWordAttributes(double[][] input_matrix, List<String> foreignWordsList) {
        List<Character> vowels = new ArrayList<>(){{
            add('a');
            add('e');
            add('i');
            add('o');
            add('u');
        }};
        List<Character> bilabials = new ArrayList<>(){{
            add('m');
            add('b');
            add('p');
        }};
        List<Character>  dentals = new ArrayList<>(){{
            add('d');
            add('t');
            add('n');
            add('c');
            add('z');
            add('s');
        }};
        List<Character>  labiodentals = new ArrayList<>(){{
            add('v');
            add('f');
        }};
        List<Character> velars = new ArrayList<>(){{
            add('k');
            add('g');
            add('h');
        }};
        List<Character> alveolars = new ArrayList<>(){{
            add('l');
            add('r');
        }};
        List<Character> palatals = new ArrayList<>(){{
            add('j');
        }};

        int input_length = foreignWordsList.size();

        for (int i = 0; i < input_length; i++) {
            int word_length = 0, vowel_count = 0, bilabials_count = 0, labiodentals_count = 0,
                    velars_count = 0, alveolars_count = 0,  palatals_count = 0, dentals_count = 0,
                    A_count = 0, E_count = 0, I_count = 0, O_count = 0, U_count = 0,
                    ascii_sum = 0, firstL_ascii = 0, lastL_ascii = 0;

            String current_word = foreignWordsList.get(i);
            word_length = current_word.length();
            for (int j = 0; j < word_length; j++) {

                ascii_sum += current_word.charAt(j);
                if (j == 0) {
                    firstL_ascii = current_word.charAt(j);
                } else if (j == current_word.length() - 1) {
                    lastL_ascii = current_word.charAt(j);
                }

                if (current_word.charAt(j) == 'a') {
                    A_count++;
                } else if (current_word.charAt(j) == 'e') {
                    E_count++;
                } else if (current_word.charAt(j) == 'i') {
                    I_count++;
                } else if (current_word.charAt(j) == 'o') {
                    O_count++;
                } else if (current_word.charAt(j) == 'u') {
                    U_count++;
                }

                if (vowels.contains(current_word.charAt(j))) {
                    if (j == 0) {
                        vowel_count += 5;
                    } else {
                        vowel_count += 1;
                    }
                } else if (bilabials.contains(current_word.charAt(j))) {
                    if (j == 0) {
                        bilabials_count += 5;
                    } else {
                        bilabials_count += 1;
                    }
                } else if (labiodentals.contains(current_word.charAt(j))) {
                    if (j == 0) {
                        labiodentals_count += 5;
                    } else {
                        labiodentals_count += 1;
                    }
                } else if (velars.contains(current_word.charAt(j))) {
                    if (j == 0) {
                        velars_count += 5;
                    } else {
                        velars_count += 1;
                    }
                } else if (alveolars.contains(current_word.charAt(j))) {
                    if (j == 0) {
                        alveolars_count += 5;
                    } else {
                        alveolars_count += 1;
                    }
                } else if (palatals.contains(current_word.charAt(j))) {
                    if (j == 0) {
                        palatals_count += 5;
                    } else {
                        palatals_count += 1;
                    }
                } else if (dentals.contains(current_word.charAt(j))) {
                    if (j == 0) {
                        dentals_count += 5;
                    } else {
                        dentals_count += 1;
                    }
                }
            }

            input_matrix[i] = new double[]{word_length, vowel_count, bilabials_count, labiodentals_count,
                    velars_count, alveolars_count, palatals_count, dentals_count,
                    A_count, E_count, I_count, O_count, U_count,
                    ascii_sum, firstL_ascii, lastL_ascii};

        }
    }
    private static void normalize(double[][] input_matrix) {
        List<Double> doubles_list = new ArrayList<>();
        double max_value = 0;
        int a = 0;
        int b = 0;

        for (int i = 0; i < input_matrix.length; i++) {
            if (a > input_matrix[i].length - 1) {
                break;
            }
            if (max_value < input_matrix[i][a]) {
                max_value = input_matrix[i][a];
            }

            if (i == input_matrix.length - 1) {
                i = 0;
                a++;
                doubles_list.add(max_value);
                max_value = 0;
            }
        }

        for (int i = 0; i < input_matrix.length; i++) {
            b = 0;
            for (int j = 0; j < input_matrix[i].length; j++) {
                if (doubles_list.get(b) != 0) {
                    input_matrix[i][j] = input_matrix[i][j] / doubles_list.get(b);
                }
                b++;
            }
        }
    }
    private static double calculateNNResults(double[] outputTest, List<String> croatianWords,
                                             List<String> foreignWordsTest) {

        int max_idx = -1;
        int idx_count = 0;
        double max_val = -10000;
        for (int i = 0; i < outputTest.length; i++) {
            if (max_val < outputTest[i]) {
                max_val = outputTest[i];
                max_idx = i;
            }
        }

        switch (max_idx) {
            case 0 -> System.out.println(foreignWordsTest.get(max_idx) +  " - Nogomet!");
            case 1 -> System.out.println(foreignWordsTest.get(max_idx) + " - Kosarka!");
            case 2 -> System.out.println(foreignWordsTest.get(max_idx) + " - Rukomet!");
            case 3 -> System.out.println(foreignWordsTest.get(max_idx) + " - Plivanje!");
            case 4 -> System.out.println(foreignWordsTest.get(max_idx) + " - Natjecanje!");
            case 5 -> System.out.println(foreignWordsTest.get(max_idx) + " - Lopta!");
            case 6 -> System.out.println(foreignWordsTest.get(max_idx) + " - Trcanje!");
            case 7 -> System.out.println(foreignWordsTest.get(max_idx) + " - Trofej!");
            case 8 -> System.out.println(foreignWordsTest.get(max_idx) + " - Medalja!");
            case 9 -> System.out.println(foreignWordsTest.get(max_idx) + " - Skijanje!");
            case 10 -> System.out.println(foreignWordsTest.get(max_idx) + " - Hokej!");
            case 11 -> System.out.println(foreignWordsTest.get(max_idx) + " - Vaterpolo!");
            default -> System.out.println("Rjesenje nije nadjeno!");
        }

        return max_idx;
//        return max_idx;
//        if (max_idx > -1 && max_idx < outputTest.length) {
//            System.out.println(foreignWordsTest.get(max_idx) + " = " + croatianWords.get(max_idx));
//            return 1;
//        } else {
//            return 0;
//        }
    }
}

