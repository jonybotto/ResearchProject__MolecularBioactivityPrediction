import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score, r2_score, roc_auc_score, root_mean_squared_error

def train_epoch(model, loader_train, optimizer, loss_fn, task='regression', use_edge_weight=True):
    model.train()
    losses = None if loss_fn is None else []

    with torch.enable_grad():
        for data in loader_train:
            optimizer.zero_grad()
            edge_weight = data.edge_weight if (hasattr(data, 'edge_weight') and use_edge_weight) else None
            y_pred = model(data.x, data.edge_index, data.batch, edge_weight=edge_weight)
            if task == 'classification':
                loss = loss_fn(y_pred.squeeze(), data.y.float().view(-1))
            else:
                loss = loss_fn(y_pred.squeeze(), data.y)

            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
        
        return np.mean(losses)

def evaluate_model(model, loader_test, loss_fn=None, task='regression', use_edge_weight=True):
    model.eval()
    y_true = []
    y_pred = []

    losses = None if loss_fn is None else []
    with torch.no_grad():
        for data in loader_test:
            y_true.extend(data.y.cpu().detach().numpy())
            edge_weight = data.edge_weight if (hasattr(data, 'edge_weight') and use_edge_weight) else None
            predictions = model(data.x, data.edge_index, data.batch, edge_weight=edge_weight)

            if task == 'classification':
                if predictions.shape[1] > 1:
                    pred_classes = predictions.argmax(dim=1)
                else:
                    pred_classes = (torch.sigmoid(predictions.view(-1)) > 0.5).long()
                y_pred.extend(pred_classes.cpu().detach().numpy().flatten())
            
            else:
                predictions = predictions.squeeze()
                y_pred.extend(predictions.cpu().detach().numpy().flatten())
                
            if loss_fn is not None:
                if task == 'classification':
                    losses.append(loss_fn(predictions.view(-1), data.y.float().view(-1)).item())
                else:
                    losses.append(loss_fn(predictions, data.y).item())
                
    if task == 'classification':
        f1 = f1_score(y_true, y_pred, average='weighted')
        if loss_fn is None:
            return {'f1': f1}
        else:
            return {'loss': np.mean(losses), 'f1': f1}
    
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        if loss_fn is None: return {'rmse': rmse}
        else: return {'loss': np.mean(losses), 'rmse': rmse}

def fit_model(model, loader_train, loader_valid, epochs, lr=0.001, patience=10, task='regression', use_edge_weight=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2)

    if task == 'classification':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    history = []
    best_val_loss = None
    no_improvement_count = 0

    for i in range(epochs):
        loss_train = train_epoch(model, loader_train, optimizer, loss_fn, task=task, use_edge_weight=use_edge_weight)
        metrics = evaluate_model(model, loader_valid, loss_fn, task=task, use_edge_weight=use_edge_weight)
        val_loss = metrics['loss']
        if best_val_loss is None:
            best_val_loss = val_loss
        else:
            if val_loss > best_val_loss:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                best_val_loss = val_loss
                torch.save(model, f'GRAPH/best_model.pt')
        if no_improvement_count >= patience:
            break
        
        scheduler.step(val_loss)
        history_entry = {
            'loss_train': loss_train,
            'loss_valid': metrics['loss'],
        }

        if task == 'classification':
            history_entry.update({
                'f1': metrics['f1']
            })
        else:
            history_entry['rmse_valid'] = metrics['rmse']
        
        history.append(history_entry)
        # print(f'Epoch {i+1}/{epochs} stats: {history[-1]}')   
    return pd.DataFrame(history)


def evaluate_test(model, loader_test, task='regression', use_edge_weight=True):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader_test:
            y_true.extend(data.y.cpu().detach().numpy())
            edge_weight = data.edge_weight if (hasattr(data, 'edge_weight') and use_edge_weight) else None
            predictions = model(data.x, data.edge_index, data.batch, edge_weight=edge_weight)

            if task == 'classification':
                if predictions.shape[1] > 1:
                    pred_classes = predictions.argmax(dim=1)
                else:
                    pred_classes = (torch.sigmoid(predictions.squeeze()) > 0.5).long()
                y_pred.extend(pred_classes.cpu().detach().numpy().flatten())
            
            else:
                predictions = predictions.squeeze()
                y_pred.extend(predictions.cpu().detach().numpy().flatten())

    if task == 'classification':   
        f1 = f1_score(y_true, y_pred)
        return {'f1': f1}
    else:
        rmse = root_mean_squared_error(y_true, y_pred)
        return {'rmse': rmse}
            

def final_evaluate(model, loader_test, task='regression', use_edge_weight=True):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for data in loader_test:
            y_true.extend(data.y.cpu().detach().numpy())
            edge_weight = data.edge_weight if (hasattr(data, 'edge_weight') and use_edge_weight) else None
            predictions = model(data.x, data.edge_index, data.batch, edge_weight=edge_weight)

            if task == 'classification':
                probs = torch.sigmoid(predictions.squeeze())
                y_scores.extend(probs.cpu().detach().numpy().flatten())
            
            else:
                predictions = predictions.squeeze()
                y_scores.extend(predictions.cpu().detach().numpy().flatten())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if task == 'classification':   
        y_pred = (y_scores > 0.5).astype(int)
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_scores': y_scores
        }
    else:
        return {
            'y_true': y_true,
            'y_pred': y_scores
        }

def predict(model, loader_predict):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for data in loader_predict:
            predictions = model(data.x, data.edge_index, data.batch).squeeze()
            y_pred.extend(predictions.cpu().detach().numpy().flatten())
    return y_pred

def final_fit_model(model, loader_train, epochs, lr=0.001, task='regression', use_edge_weight=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if task == 'classification':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    history = []
    for i in range(epochs):
        loss_train = train_epoch(model, loader_train, optimizer, loss_fn, task=task, use_edge_weight=use_edge_weight)
       
        history_entry = {
            'loss_train': loss_train,
        }

        history.append(history_entry)
        # print(f'Epoch {i+1}/{epochs} stats: {history[-1]}')   
    return pd.DataFrame(history)