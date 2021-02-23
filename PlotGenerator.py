import matplotlib.pyplot as plt

if __name__ == "__plot_T_DP_graph__":
    plot_T_DP_graph()

if __name__ == "__plot_labels_graph__":
    plot_labels_graph()

def plot_T_DP_graph (pred_output, predictions_df, y_inputs, y_test_data, fig_save_path):

    # THIS FUNCTION PLOTS SPECIFICALLY THE PREDICTED TEMPERATURE AND DEW POINT FOR COMPARISON BETWEEN THE TWO

    plt.plot(predictions_df[pred_output[0]], color = 'black', label = pred_output[0])
    plt.plot(predictions_df[pred_output[1]], color = 'pink', label = pred_output[1])
    plt.plot(y_test_data[y_inputs[0]], color = 'yellow', label = y_inputs[0])
    plt.plot(y_test_data[y_inputs[1]], color = 'red', label = y_inputs[1])
    plt.axvspan(y_test_data.index.min(), y_test_data.index.max(), facecolor='black', alpha=0.15) #Plots a grey area which corresponds to real values
    plt.xlabel('Time [Day / Hour]')
    plt.ylabel('Temperature ºC')
    plt.legend()
    plt.savefig (fig_save_path, dpi = 2030)
    plt.show()

def plot_labels_graph (pred_output, predictions_df, y_inputs, y_test_data):
    
    for column in range(len(pred_output)):

        plt.plot(predictions_df[pred_output[column]], color = 'black', label = '{}'.format(pred_output[column]))
        plt.plot(y_test_data[y_inputs[column]], color = 'yellow', label =  '{}'.format(y_inputs[column]))
        plt.axvspan(y_test_data.index.min(), y_test_data.index.max(), facecolor='black', alpha=0.15) #Plots a grey area which corresponds to real values
        plt.xlabel('Time [Day / Hour]')
        plt.ylabel('Prediction')
        plt.legend()
        plt.show()


    