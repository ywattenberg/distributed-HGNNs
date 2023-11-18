#include "scores.h"

torch::Tensor accuracy(torch::Tensor &y_pred, torch::Tensor &y_true){
    torch::Tensor correct = torch::zeros({1}, torch::kLong);
    torch::Tensor total = torch::zeros({1}, torch::kLong);
    torch::Tensor y_pred_max = torch::argmax(y_pred, 1);
    torch::Tensor cmp = torch::eq(y_pred_max, y_true);
    correct += torch::sum(cmp);
    total += y_true.size(0);
    return correct.div(total);
}

torch::Tensor precision(torch::Tensor &y_pred, torch::Tensor &y_true, int label){
    torch::Tensor true_positives = torch::zeros({1}, torch::kLong);
    torch::Tensor total_positives = torch::zeros({1}, torch::kLong);
    torch::Tensor y_pred_max = torch::argmax(y_pred, 1);
    torch::Tensor cmp = torch::eq(y_pred_max, y_true);
    torch::Tensor elems_of_class = y_true.eq(label);
    // std::cout << "total elements of this label: " << torch::sum(elems_of_class) << std::endl;
    true_positives += torch::dot(cmp.to(torch::kInt), elems_of_class.to(torch::kInt)).item<long>();
    total_positives += torch::sum(y_pred_max.eq(label)).item<long>();

    if (total_positives.item<long>() == 0){
        return total_positives;
    } else {
       return true_positives.div(total_positives);
    }
    
    
}


double multiclass_precision(torch::Tensor &y_pred, torch::Tensor &y_true, int num_classes){
    // torch::Tensor sum = torch::zeros({1}, torch::kDouble);
    double sum = 0.0;
    for (int i = 0; i < num_classes; i++){
        sum += precision(y_pred, y_true, i).item<double>();
    }
    return sum / num_classes;
}

torch::Tensor recall(torch::Tensor &y_pred, torch::Tensor &y_true, int label){
    torch::Tensor true_positives = torch::zeros({1}, torch::kLong);
    torch::Tensor actual_positives = torch::zeros({1}, torch::kLong);
    torch::Tensor y_pred_max = torch::argmax(y_pred, 1);
    torch::Tensor cmp = torch::eq(y_pred_max, y_true);
    torch::Tensor elems_of_class = y_true.eq(label);
    true_positives += torch::dot(cmp.to(torch::kInt), elems_of_class.to(torch::kInt)).item<long>();
    actual_positives += torch::sum(elems_of_class).item<long>();
    return true_positives.div(actual_positives);
}

double multiclass_recall(torch::Tensor &y_pred, torch::Tensor &y_true, int num_classes){
    // torch::Tensor sum = torch::zeros({1}, torch::kDouble);
    double sum = 0.0;
    for (int i = 0; i < num_classes; i++){
        sum += recall(y_pred, y_true, i).item<double>();
    }
    return sum / num_classes;
}

torch::Tensor f1_score(torch::Tensor &y_pred, torch::Tensor &y_true, int num_classes){
    torch::Tensor f1 = torch::zeros({1}, torch::kDouble);
    double prec = multiclass_precision(y_pred, y_true, num_classes);
    std::cout << "multiclass precision was: " << prec << std::endl;
    double rec = multiclass_recall(y_pred, y_true, num_classes);
    f1 +=  (2 * prec * rec / (prec + rec));
    return f1;
}

