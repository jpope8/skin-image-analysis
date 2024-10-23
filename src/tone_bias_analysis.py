import matplotlib.pyplot as plt
import sys
import json
import os
import pathlib

def get_files( dir_path, ext ):
    files = list()
    for file in os.listdir(dir_path):
        if file.endswith(ext):
            files.append(file)
    return files


def print_epoch_results(epoch_results):
    # Some information about specific epoch

    tone_di_results = epoch_results['tone_di_results']
    print(f"    keys {epoch_results.keys()}")
    print(f"    accuracy {epoch_results['accuracy']}")
    print(f"    tone_di {epoch_results['tone_di']}")
    print(f"    tone_di_results {tone_di_results.keys()}")
    print("     TONE DI RESULTS")
    for key in tone_di_results.keys():
        value = tone_di_results[key]
        print(f"        [{key}] -> {value}")
    pos_min = tone_di_results['tp_min'] + tone_di_results['fp_min']
    min_count = tone_di_results['min_count']
    selection_rate_min = pos_min/min_count
    print(f"    selection_rate_min = {pos_min}/{min_count} = {selection_rate_min:.4f}")
    pos_maj = tone_di_results['tp_maj'] + tone_di_results['fp_maj']
    maj_count = tone_di_results['maj_count']
    selection_rate_maj = pos_maj / maj_count
    print(f"    selection_rate_maj = {pos_maj}/{maj_count} = {selection_rate_maj:.4f}")
    tone_di = selection_rate_min / selection_rate_maj
    print(f"    tone_di = {selection_rate_min:.4f}/{selection_rate_maj:.4f} = {tone_di:.4f}")



def main_old():
    #  Get some cli arguments for preprocessing, training, evaluation.
    if len(sys.argv) != 3:
        print(f"Usage: <folder with JSON files> <epoch | all>")
        print(f"Example: ./results/balanced_2024-09-16_17-26-21/")
        return
    dir_path = sys.argv[1]
    epoch_max = sys.argv[2]



    # Find all file with json extension
    json_files = get_files( dir_path, '.json' )
    #print(f"BEFORE SORT: {json_files}")
    # Sort to ensure training plots are in datatime order (datetime is in name of json file)
    json_files.sort()
    #print(f"AFTER SORT: {json_files}")

    train_losses = list()
    val_accuracies = list()
    train_accuracies = list()
    tone_dis = list()
    gender_dis = list()
    control_dis = list()
    f1_scores = list()

    epochs = list()

    # Assume first file has 1st epoch
    global_epoch = 1
    stop = False
    for json_path_name in json_files:
        json_path_name = os.path.join( dir_path, json_path_name )
        with open(json_path_name, "r") as json_file:
            for line in json_file:
                results = json.loads(line)
                #print(f"{results}")
                epoch = results['epoch']
                val_accuracy = results['accuracy']
                train_accuracy = results['train_accuracy']
                train_loss = results['avg_batch_loss']

                tone_di_results = results['tone_di_results']
                gender_di_results = results['gender_di_results']
                control_di_results = results['control_di_results']
                tone_di = tone_di_results['di']
                gender_di = gender_di_results['di']
                control_di = control_di_results['di']

                f1_score = tone_di_results['f1']

                # sanity check
                if epoch > global_epoch:
                    raise ValueError(f"Unexpected epoch {epoch}, greater than {global_epoch}")

                epochs.append(global_epoch)
                train_losses.append(train_loss)
                val_accuracies.append(val_accuracy)
                train_accuracies.append(train_accuracy)
                tone_dis.append(tone_di)
                gender_dis.append(gender_di)
                control_dis.append(control_di)
                f1_scores.append(f1_score)

                global_epoch += 1

                if epoch_max != "all" and global_epoch == int(epoch_max):
                    stop = True
                    break

        if stop:
            break

    # Set default figure size to 10 inches wide and 6 inches tall
    scale = 2.0
    #plt.rcParams['figure.figsize'] = (7*scale, 4*scale)
    fig_width  = 7*scale
    fig_height = 4*scale
    fig, plot_di = plt.subplots(figsize=(fig_width, fig_height))

    # Twin, like two plots but in same space, want loss and disparate impact on same plot
    plot_loss = plot_di.twinx()

    symbol_size = 4

    # ======================================================================= #
    # For accuracy plots
    # ======================================================================= #
    # # plot_di.plot(epochs, f1_scores, 'b', label='F1 score')
    # #plot_loss.scatter(epochs, train_losses, marker='s', s=symbol_size, c='g', label='Training Loss')
    # #plot_di.scatter(epochs, val_accuracies, marker='o', s=symbol_size, c='r', label='Validation Accuracy')
    # #plot_di.scatter(epochs, train_accuracies, marker='^', s=symbol_size, c='b', label='Train Accuracy')
    #
    # plot_loss.plot(epochs, train_losses, marker='s', markersize=symbol_size, c='g', label='Training Loss')
    # plot_di.plot(epochs, val_accuracies, marker='o', markersize=symbol_size, c='r', label='Validation Accuracy')
    # plot_di.plot(epochs, train_accuracies, marker='^', markersize=symbol_size, c='b', label='Train Accuracy')
    #
    # # # Draw a horizontal line for majority classifier
    # y_value = 0.74 # for imbalanced 0.74, for balanced, it depends but usually about 0.55
    # plot_di.axhline(y=y_value, xmin=0, xmax=global_epoch, color='black', linestyle='dashed', linewidth=1)
    # plot_di.text( global_epoch*0.5, y_value+0.01, "Majority Classifier", fontsize=10, color='black')
    #
    # max_y = 1.0
    # y_label = 'Accuracy'

    # ======================================================================= #
    # For DI plots
    # ======================================================================= #
    plot_loss.plot(epochs, train_losses, marker='s', markersize=symbol_size, color='g', label='Training Loss')

    plot_di.plot(epochs, tone_dis, marker='o', markersize=symbol_size, color='r', label='Tone Disparate Impact')
    #plot_di.plot(epochs, gender_dis, color='b', label='Gender Disparate Impact')
    plot_di.plot(epochs, control_dis, marker='^', markersize=symbol_size, color='b', label='Control Disparate Impact')

    # Draw a horizontal line and text for expected di
    plot_di.axhline(y=1.2, xmin=0, xmax=global_epoch, color='black', linestyle='dashed', linewidth=1)
    plot_di.text(global_epoch*0.5, 1.22, "Biased DI", fontsize=10, color='black')
    plot_di.text(global_epoch*0.5, 1.15, "Unbias DI", fontsize=10, color='black')

    # Draw a horizontal line and text for biased di
    plot_di.axhline(y=0.80, xmin=0, xmax=global_epoch, color='black', linestyle='dashed', linewidth=1)
    plot_di.text(global_epoch*0.5, 0.82, "Unbias DI", fontsize=10, color='black')
    plot_di.text(global_epoch*0.5, 0.76, "Biased DI", fontsize=10, color='black')
    max_y = 1.3
    y_label = 'Disparate Impact (DI)'

    # ======================================================================= #
    # For ALL plots
    # ======================================================================= #
    # Some constants, restrict max DI/Loss, both always greater than 0
    min_y = 0

    plot_di.set_ylim(min_y, max_y)
    plot_loss.set_ylim(min_y, max_y)
    # Constant, need room for text describing the thresholds
    plot_di.set_xlim(-10, int(global_epoch * 1.05))

    #plt.ylim( 0, 1.3 )
    #plt.xlim(-10, int(global_epoch*1.05))

    #plt.title('Training and Validation Loss')
    plot_di.set_xlabel('Epoch')
    plot_di.set_ylabel(y_label)
    plot_loss.set_ylabel('Loss')

    # Each plot has its own legend
    plot_di.legend(loc="lower left", framealpha=1.0)
    plot_loss.legend(loc="lower right", framealpha=1.0)


    # Get the folder's name so we can use in name of mng to save
    path = pathlib.Path(dir_path)
    #print(f"NAME {path.name}")  # Output: /path/to/folder/subfolder
    #my_dpi = 96 # not sure this is necessary and may not be portable
    #plt.savefig('my_fig.png', dpi=my_dpi)
    plt.savefig(f'figure_{path.name}.png')

    plt.show()



def process_single_json():
    #  Get some cli arguments for preprocessing, training, evaluation.
    if len(sys.argv) != 2:
        print(f"Usage: <JSON file with results>")
        print(f"Example: ./results/balanced/2024-09-16_17-26-21.json")
        return
    json_path_name = sys.argv[1]  # root_dir_name = "./tone"

    train_losses = list()
    val_accuracies = list()
    disparate_impacts = list()
    epochs = list()
    with open( json_path_name, "r" ) as json_file:
        e = 0
        for line in json_file:
            results = json.loads( line )
            #print(f"{results}")
            epoch = results['epoch']
            val_accuracy = results['accuracy']
            train_loss = results['avg_batch_loss']

            tone_di_results = results['tone_di_results']
            di = tone_di_results['di']
            if e != epoch:
                raise ValueError(f"Unexpected epoch {epoch}, expected {e}")

            epochs.append(e)
            train_losses.append(train_loss)
            val_accuracies.append(val_accuracy)
            disparate_impacts.append(di)
            e += 1


    plt.plot(epochs, train_losses, 'g', label='Training Loss')
    plt.plot(epochs, val_accuracies, 'b', label='Validation Accuracy')
    plt.plot(epochs, disparate_impacts, 'r', label='Disparate Impact')

    plt.title('Disparate Impact and Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def transpose_dict(results_dict):
    """
    [global_epoch] -> {results}
    [1] -> {'epoch': 1, 'accuracy': 0.55, ...}
    [2] -> {'epoch': 2, 'accuracy': 0.55, ...}
    [3] -> {'epoch': 3, 'accuracy': 0.56, ...}
    [4] -> {'epoch': 4, 'accuracy': 0.56, ...}
    [5] -> {'epoch': 5, 'accuracy': 0.57, ...}

    returns
    ['accuracy'] -> [0.55,0.55,0.56,0.56,0.57]
    ['epoch']    -> [   1,   2,   3,   4,   5]
    """
    transposed_dict = dict()
    # Assume epoch 1 has all keys
    header_results = results_dict[1]
    for measure_name in header_results.keys():
        measure_values = get_measure(results_dict, measure_name)
        transposed_dict[measure_name] = measure_values
    return transposed_dict

def get_measure(results_dict, measure_name):
    """
    [global_epoch] -> {results}
    [1] -> {'epoch': 1, 'accuracy': 0.55, ...}
    [2] -> {'epoch': 2, 'accuracy': 0.55, ...}
    [3] -> {'epoch': 3, 'accuracy': 0.56, ...}
    [4] -> {'epoch': 4, 'accuracy': 0.56, ...}
    [5] -> {'epoch': 5, 'accuracy': 0.57, ...}

    returns
    [0.55,0.55,0.56,0.56,0.57]
    """
    measure_values = list()
    for global_epoch in results_dict.keys():
        epoch_results = results_dict[global_epoch]
        measure_value = epoch_results[measure_name]
        measure_values.append(measure_value)
    return measure_values




def read_experiment( exp_path ):
    # Find all file with json extension
    json_files = get_files(exp_path, '.json')
    # print(f"BEFORE SORT: {json_files}")
    # Sort to ensure training plots are in datatime order (datetime is in name of json file)
    json_files.sort()
    # print(f"AFTER SORT: {json_files}")

    experiment_results = dict()

    # Assume first file has 1st epoch
    global_epoch = 1
    stop = False
    for json_path_name in json_files:
        json_path_name = os.path.join(exp_path, json_path_name)
        with open(json_path_name, "r") as json_file:
            for line in json_file:
                """
                We want to map the global epoch to a dict of results for that epoch
                but we already have those results as a dict ;)
                So just need to map, noting that the global_epoch != epoch
                
                [global_epoch] -> {results}
                [1] -> {'epoch': 1, 'accuracy': 0.55, ...}  File 2024-09-27_08-29-36.json
                [2] -> {'epoch': 2, 'accuracy': 0.55, ...}
                [3] -> {'epoch': 1, 'accuracy': 0.56, ...}  File 2024-09-28_22-39-07.json
                [4] -> {'epoch': 2, 'accuracy': 0.56, ...}
                [5] -> {'epoch': 3, 'accuracy': 0.57, ...}
                """
                results = json.loads(line)

                # We have a dict of dicts making life hard, flatten
                results['tone_di'] = results['tone_di_results']['di']
                # why f1 for tone, classifier diagnosis?  Bad approach, f1 is same for all di results
                results['f1'] = results['tone_di_results']['f1']

                results['gender_di'] = results['gender_di_results']['di']
                results['control_di'] = results['control_di_results']['di']

                # Sanity check, will no longer need the epoch
                epoch = results['epoch']
                if epoch > global_epoch:
                    raise ValueError(f"Unexpected epoch {epoch}, greater than {global_epoch}")

                # Debatable, overwrite so epoch in value dict is same as global
                results['epoch'] = global_epoch

                """
                [global_epoch] -> {results}
                [1] -> {'epoch': 1, 'accuracy': 0.55, ...}
                [2] -> {'epoch': 2, 'accuracy': 0.55, ...}
                [3] -> {'epoch': 3, 'accuracy': 0.56, ...}
                [4] -> {'epoch': 4, 'accuracy': 0.56, ...}
                [5] -> {'epoch': 5, 'accuracy': 0.57, ...}
                """
                experiment_results[global_epoch] = results
                global_epoch += 1

    return experiment_results

def read_experiments(experiments_folder, prefix, epoch_to_detail):
    #experiments_folder = "results"
    #prefix = "imbalanced"
    # First get list of the folders with the specified prefix
    files = list()
    for file in os.listdir(experiments_folder):
        if file.startswith(prefix):
            files.append(file)

    experiments = dict()
    for experiment_folder in files:
        exp_path = os.path.join(experiments_folder, experiment_folder)
        #print(f"{exp_path}")
        experiment_results = read_experiment(exp_path)
        experiments[exp_path] = experiment_results

    """
    balanced_2024-09-21_00-38-39
    [1] -> {'epoch': 1, 'accuracy': 0.55, ...}
    [2] -> {'epoch': 2, 'accuracy': 0.55, ...}
    [3] -> {'epoch': 3, 'accuracy': 0.56, ...}
    [4] -> {'epoch': 4, 'accuracy': 0.56, ...}
    [5] -> {'epoch': 5, 'accuracy': 0.57, ...}

    balanced_2024-09-21_10-18-46
    [1] -> {'epoch': 1, 'accuracy': 0.65, ...}
    [2] -> {'epoch': 2, 'accuracy': 0.65, ...}
    [3] -> {'epoch': 3, 'accuracy': 0.66, ...}
    [4] -> {'epoch': 4, 'accuracy': 0.66, ...}
    [5] -> {'epoch': 5, 'accuracy': 0.67, ...}

    :returns
    [1] -> {'epoch': 1, 'accuracy': 0.60, ...} average
    [2] -> {'epoch': 2, 'accuracy': 0.60, ...} average
    [3] -> {'epoch': 3, 'accuracy': 0.61, ...} average
    [4] -> {'epoch': 4, 'accuracy': 0.61, ...} average
    [5] -> {'epoch': 5, 'accuracy': 0.62, ...} average
    """
    experiment_accumulator = dict()
    counts = dict()
    for exp_path in experiments.keys():
        experiment = experiments[exp_path]
        print(f"FILE {exp_path} epochs {len(experiment)}")
        for epoch in experiment.keys():
            # Add epoch if not already
            if epoch not in experiment_accumulator:
                experiment_accumulator[epoch] = dict()
            accumulator_epoch = experiment_accumulator[epoch]

            epoch_results = experiment[epoch] # epoch_results is a dict
            #print(f"[{epoch}] -> {epoch_results.keys()}")
            for measure_name in epoch_results.keys():
                # Some values are not numbers, some are dict
                measure_value = epoch_results[measure_name]
                if not isinstance(measure_value, dict):
                    if measure_name not in accumulator_epoch:
                        accumulator_epoch[measure_name] = 0.0
                        counts[f"{epoch}_{measure_name}_count"] = 0
                    accumulator_epoch[measure_name] += measure_value
                    counts[f"{epoch}_{measure_name}_count"] += 1

            # DEBUGGING, no effect on processing
            #if epoch == epoch_to_detail:
            #     print(f"EPOCH DETAILS {epoch_to_detail} experiment {exp_path}")
            #     print_epoch_results(epoch_results)

    #for epoch in range(1, epoch_to_detail):
    for epoch in experiment_accumulator.keys():
        epoch_results = experiment_accumulator[epoch]
        #print(f"EPOCH {epoch}")
        for measure_name in epoch_results.keys():
            measure_value = epoch_results[measure_name]
            measure_count = counts[f"{epoch}_{measure_name}_count"]
            measure_avg = measure_value/measure_count
            #print(f"    {measure_name} -> {measure_value}   count {measure_count}  avg={measure_avg:.2f}")
            epoch_results[measure_name] = measure_avg


    # # Some information about the last epoch
    # epoch_results = experiment_accumulator[epoch_to_detail]
    # print(f"EPOCH DETAILS {epoch_to_detail}")
    # print(f"accuracy {epoch_results['accuracy']}")
    # print(f"keys {epoch_results.keys()}")
    # print(f"tone_di {epoch_results['tone_di']}")

    return experiment_accumulator


def main():
    #  Get some cli arguments for preprocessing, training, evaluation.
    if len(sys.argv) != 3:
        print(f"Usage: <folder with JSON files> <epoch_to_detail>")
        print(f"Example: ./results/balanced_2024-09-16_17-26-21/")
        return
    exp_path = sys.argv[1]
    epoch_to_detail = int(sys.argv[2]) # provide more detail about this epoch's results

    #experiment_results = read_experiment(exp_path)
    experiment_results = read_experiments("results", exp_path, epoch_to_detail)

    """
    print(f"\n\n##### RESULTS")
    for epoch in experiment_results.keys():
        epoch_results = experiment_results[epoch]
        print(f"    [{epoch}] - > {epoch_results}")
    """

    results_lists = transpose_dict(experiment_results)

    #print(f"{results_lists.keys()}")

    epochs         = results_lists["epoch"]
    train_losses   = results_lists["avg_batch_loss"]
    val_accuracies = results_lists["accuracy"]
    train_accuracies = results_lists["train_accuracy"]
    tone_dis       = results_lists["tone_di"]
    gender_dis     = results_lists["gender_di"]
    control_dis    = results_lists["control_di"]
    f1_scores      = results_lists["f1"]

    global_epoch = len(epochs)


    # Set default figure size to 10 inches wide and 6 inches tall
    scale = 2.0
    #plt.rcParams['figure.figsize'] = (7*scale, 4*scale)
    fig_width  = 7*scale
    fig_height = 4*scale
    fig, plot_di = plt.subplots(figsize=(fig_width, fig_height))

    # Twin, like two plots but in same space, want loss and disparate impact on same plot
    plot_loss = plot_di.twinx()

    symbol_size = 4

    # ======================================================================= #
    # For accuracy plots
    # ======================================================================= #
    # # plot_di.plot(epochs, f1_scores, 'b', label='F1 score')
    # #plot_loss.scatter(epochs, train_losses, marker='s', s=symbol_size, c='g', label='Training Loss')
    # #plot_di.scatter(epochs, val_accuracies, marker='o', s=symbol_size, c='r', label='Validation Accuracy')
    # #plot_di.scatter(epochs, train_accuracies, marker='^', s=symbol_size, c='b', label='Train Accuracy')
    #
    # plot_loss.plot(epochs, train_losses, marker='s', markersize=symbol_size, c='g', label='Training Loss')
    # plot_di.plot(epochs, val_accuracies, marker='o', markersize=symbol_size, c='r', label='Validation Accuracy')
    # plot_di.plot(epochs, train_accuracies, marker='^', markersize=symbol_size, c='b', label='Train Accuracy')
    #
    # # # Draw a horizontal line for majority classifier
    # y_value = 0.74 # for imbalanced 0.74, for balanced, it depends but usually about 0.55
    # plot_di.axhline(y=y_value, xmin=0, xmax=global_epoch, color='black', linestyle='dashed', linewidth=1)
    # plot_di.text( global_epoch*0.5, y_value+0.01, "Majority Classifier", fontsize=10, color='black')
    #
    # max_y = 1.0
    # y_label = 'Accuracy'

    # ======================================================================= #
    # For DI plots
    # ======================================================================= #
    plot_loss.plot(epochs, train_losses, marker='s', markersize=symbol_size, color='g', label='Training Loss')

    plot_di.plot(epochs, tone_dis, marker='o', markersize=symbol_size, color='r', label='Tone Disparate Impact')
    #plot_di.plot(epochs, gender_dis, color='orange', label='Gender Disparate Impact')
    plot_di.plot(epochs, control_dis, marker='^', markersize=symbol_size, color='b', label='Control Disparate Impact')

    # Draw a horizontal line and text for expected di
    plot_di.axhline(y=1.2, xmin=0, xmax=global_epoch, color='black', linestyle='dashed', linewidth=1)
    plot_di.text(global_epoch*0.5, 1.22, "Biased DI", fontsize=10, color='black')
    plot_di.text(global_epoch*0.5, 1.15, "Unbias DI", fontsize=10, color='black')

    # Draw a horizontal line and text for biased di
    plot_di.axhline(y=0.80, xmin=0, xmax=global_epoch, color='black', linestyle='dashed', linewidth=1)
    plot_di.text(global_epoch*0.5, 0.82, "Unbias DI", fontsize=10, color='black')
    plot_di.text(global_epoch*0.5, 0.76, "Biased DI", fontsize=10, color='black')
    max_y = 1.3
    y_label = 'Disparate Impact (DI)'

    # ======================================================================= #
    # For ALL plots
    # ======================================================================= #
    # Some constants, restrict max DI/Loss, both always greater than 0
    min_y = 0

    plot_di.set_ylim(min_y, max_y)
    plot_loss.set_ylim(min_y, max_y)
    # Constant, need room for text describing the thresholds
    plot_di.set_xlim(-10, int(global_epoch * 1.05))

    #plt.ylim( 0, 1.3 )
    #plt.xlim(-10, int(global_epoch*1.05))

    #plt.title('Training and Validation Loss')
    plot_di.set_xlabel('Epoch')
    plot_di.set_ylabel(y_label)
    plot_loss.set_ylabel('Loss')

    # Each plot has its own legend
    plot_di.legend(loc="lower left", framealpha=1.0)
    plot_loss.legend(loc="lower right", framealpha=1.0)


    # Get the folder's name so we can use in name of mng to save
    path = pathlib.Path(exp_path)
    #print(f"NAME {path.name}")  # Output: /path/to/folder/subfolder
    #my_dpi = 96 # not sure this is necessary and may not be portable
    #plt.savefig('my_fig.png', dpi=my_dpi)
    plt.savefig(f'figure_{path.name}.png')

    plt.show()

if __name__ == '__main__':
    main()