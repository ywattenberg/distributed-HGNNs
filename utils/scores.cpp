#include "scores.h"

torch::Tensor accuracy(torch::Tensor &y_pred, torch::Tensor &y_true){

    torch::Tensor correct = torch::zeros({1}, torch::kLong);
    torch::Tensor total = torch::zeros({1}, torch::kLong);
    torch::Tensor y_pred_max = torch::argmax(y_pred, 1);
    torch::Tensor cmp = torch::eq(y_pred_max, y_true);
    correct += torch::sum(cmp).item<long>();
    total += y_true.size(0);
    return correct.div(total);
}

torch::Tensor precision(torch::Tensor &y_pred, torch::Tensor &y_true){
    torch::Tensor true_positives = torch::zeros({1}, torch::kLong);
    torch::Tensor predicted_positives = torch::zeros({1}, torch::kLong);
    torch::Tensor y_pred_max = torch::argmax(y_pred, 1);
    torch::Tensor cmp = torch::eq(y_pred_max, y_true);
    true_positives += torch::sum(cmp).item<long>();
    predicted_positives += torch::sum(y_pred_max).item<long>();
    return true_positives.div(predicted_positives);
}

torch::Tensor recall(torch::Tensor &y_pred, torch::Tensor &y_true){
    torch::Tensor true_positives = torch::zeros({1}, torch::kLong);
    torch::Tensor actual_positives = torch::zeros({1}, torch::kLong);
    torch::Tensor y_pred_max = torch::argmax(y_pred, 1);
    torch::Tensor cmp = torch::eq(y_pred_max, y_true);
    true_positives += torch::sum(cmp).item<long>();
    actual_positives += torch::sum(y_true).item<long>();
    return true_positives.div(actual_positives);
}

torch::Tensor f1_score(torch::Tensor &y_pred, torch::Tensor &y_true){
    torch::Tensor prec = precision(y_pred, y_true);
    torch::Tensor rec = recall(y_pred, y_true);
    return 2 * prec * rec / (prec + rec);
}

