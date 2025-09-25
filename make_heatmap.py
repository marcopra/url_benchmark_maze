#!/usr/bin/env python3
"""
Script per generare heatmap degli stati fisici dal dataset point_mass_maze.
Riproduce le visualizzazioni mostrate nell'immagine allegata.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from pathlib import Path
from replay_buffer import load_episode
try:
    from scipy.stats import winsorize
except ImportError:
    print("Warning: scipy non disponibile, winsorize non funzionerà")
    winsorize = None


def collect_states_from_dataset(replay_dir):
    """Carica tutti gli stati fisici dal dataset."""
    replay_path = Path(replay_dir)
    episode_files = list(replay_path.glob('*.npz'))
    
    if not episode_files:
        raise ValueError(f"Nessun file episodio trovato in {replay_dir}")
    
    print(f"Trovati {len(episode_files)} file episodio")
    
    all_states = []
    total_episode_states = 0
    
    for i, eps_fn in enumerate(episode_files):
        try:
            episode = load_episode(eps_fn)
            # print(f"Episodio {i+1}/{len(episode_files)}: {eps_fn.name}")
            # print(f"  Chiavi disponibili: {list(episode.keys())}")
            
            if 'physics' in episode:
                states = episode['physics']
            elif 'proprio_observation' in episode:
                states = episode['proprio_observation']
                if states.shape[1] > 2:
                    states = states[:, :2]  # Limita a 2 dimensioni

            else:
                print(f"  Avvertimento: 'physics' non trovato, keys disponibili: {list(episode.keys())}")
            all_states.append(states)
            total_episode_states += states.shape[0]
        except Exception as e:
            print(f"  Errore nel caricamento: {e}")
            continue
    
    if all_states:
        # Concatena tutti gli stati
        all_states = np.concatenate(all_states, axis=0)
        print(f"\n=== SOMMARIO CARICAMENTO ===")
        print(f"Episodi processati: {len(episode_files)}")
        print(f"Episodi caricati con successo: {len(all_states)}")
        print(f"Stati totali raccolti: {all_states.shape[0]}")
        print(f"Verifica conteggio: {total_episode_states}")
        print(f"Dimensione stato: {all_states.shape[1:]}")
        
        if all_states.shape[1] >= 2:
            print(f"Range finale X: [{all_states[:, 0].min():.4f}, {all_states[:, 0].max():.4f}]")
            print(f"Range finale Y: [{all_states[:, 1].min():.4f}, {all_states[:, 1].max():.4f}]")
        
        return all_states
    else:
        raise ValueError("Nessuno stato caricato dal dataset")


def preprocess_states(states, method='none', percentile=95, std_factor=3):
    """Applica preprocessing agli stati per migliorare la visualizzazione."""
    x, y = states[:, 0], states[:, 1]
    original_count = len(x)
    
    if method == 'none':
        return x, y, original_count
    
    elif method == 'clip_percentile':
        # Clippa agli outlier usando percentili
        x_low, x_high = np.percentile(x, [100-percentile, percentile])
        y_low, y_high = np.percentile(y, [100-percentile, percentile])
        
        x_clipped = np.clip(x, x_low, x_high)
        y_clipped = np.clip(y, y_low, y_high)
        
        print(f"Clipping percentile {percentile}%:")
        print(f"  X: [{x_low:.4f}, {x_high:.4f}]")
        print(f"  Y: [{y_low:.4f}, {y_high:.4f}]")
        
        return x_clipped, y_clipped, original_count
    
    elif method == 'remove_outliers':
        # Rimuove outlier usando la regola 3-sigma o percentili
        x_mean, x_std = x.mean(), x.std()
        y_mean, y_std = y.mean(), y.std()
        
        x_mask = np.abs(x - x_mean) <= std_factor * x_std
        y_mask = np.abs(y - y_mean) <= std_factor * y_std
        mask = x_mask & y_mask
        
        x_filtered = x[mask]
        y_filtered = y[mask]
        
        print(f"Rimozione outliers (±{std_factor}):")
        print(f"  Stati rimossi: {original_count - len(x_filtered)} ({(original_count - len(x_filtered))/original_count*100:.2f}%)")
        print(f"  Range X: [{x_filtered.min():.4f}, {x_filtered.max():.4f}]")
        print(f"  Range Y: [{y_filtered.min():.4f}, {y_filtered.max():.4f}]")
        
        return x_filtered, y_filtered, len(x_filtered)
    
    elif method == 'winsorize':
        # Winsorizzazione: sostituisce outlier con valori percentili
        x_wins = winsorize(x, limits=[(100-percentile)/100, (100-percentile)/100])
        y_wins = winsorize(y, limits=[(100-percentile)/100, (100-percentile)/100])
        
        print(f"Winsorizzazione al {percentile}%:")
        print(f"  Range X: [{x_wins.min():.4f}, {x_wins.max():.4f}]")
        print(f"  Range Y: [{y_wins.min():.4f}, {y_wins.max():.4f}]")
        
        return x_wins, y_wins, original_count
    
    elif method == 'log_transform':
        # Trasformazione logaritmica (sposta tutti i valori in positivo prima)
        x_shifted = x - x.min() + 1e-8
        y_shifted = y - y.min() + 1e-8
        
        x_log = np.log(x_shifted)
        y_log = np.log(y_shifted)
        
        print(f"Trasformazione logaritmica:")
        print(f"  Range X log: [{x_log.min():.4f}, {x_log.max():.4f}]")
        print(f"  Range Y log: [{y_log.min():.4f}, {y_log.max():.4f}]")
        
        return x_log, y_log, original_count
    
    elif method == 'robust_scale':
        # Scaling robusto usando mediana e IQR
        x_median, x_iqr = np.median(x), np.percentile(x, 75) - np.percentile(x, 25)
        y_median, y_iqr = np.median(y), np.percentile(y, 75) - np.percentile(y, 25)
        
        x_scaled = (x - x_median) / (x_iqr + 1e-8)
        y_scaled = (y - y_median) / (y_iqr + 1e-8)
        
        print(f"Scaling robusto:")
        print(f"  Range X scaled: [{x_scaled.min():.4f}, {x_scaled.max():.4f}]")
        print(f"  Range Y scaled: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")
        
        return x_scaled, y_scaled, original_count
    
    else:
        raise ValueError(f"Metodo preprocessing sconosciuto: {method}")


def create_heatmap(states, title="Heatmap degli Stati", bins=50, figsize=(8, 6), 
                  preprocessing='none', percentile=95, std_factor=3, colormap='Blues',
                  vmin_percentile=10, vmax_percentile=90, xlim=None, ylim=None, use_counts=False):
    """Crea una heatmap 2D degli stati."""
    if states.shape[1] < 2:
        raise ValueError("Gli stati devono avere almeno 2 dimensioni per la heatmap 2D")
    
    print(f"\n=== PREPROCESSING: {preprocessing} ===")
    x, y, final_count = preprocess_states(states, preprocessing, percentile, std_factor)
    
    print(f"\n=== DEBUG HEATMAP ===")
    print(f"Numero di punti originali: {len(states)}")
    print(f"Numero di punti dopo preprocessing: {final_count}")
    print(f"Range X finale: [{x.min():.6f}, {x.max():.6f}]")
    print(f"Range Y finale: [{y.min():.6f}, {y.max():.6f}]")
    print(f"Bins utilizzati: {bins}")
    
    # Verifica se i dati sono concentrati in un'area specifica
    x_unique = len(np.unique(x))
    y_unique = len(np.unique(y))
    print(f"Valori unici X: {x_unique}")
    print(f"Valori unici Y: {y_unique}")
    
    # Mostra distribuzione dei valori
    print(f"Media X: {x.mean():.6f}, Std X: {x.std():.6f}")
    print(f"Media Y: {y.mean():.6f}, Std Y: {y.std():.6f}")
    
    # Crea la figura
    plt.figure(figsize=figsize)
    
    # Imposta i limiti per l'istogramma se specificati
    if xlim is not None and ylim is not None:
        print(f"Usando limiti fissi: X={xlim}, Y={ylim}")
        range_limits = [xlim, ylim]
    else:
        range_limits = None
    
    # Calcola i limiti della colormap basati sui percentili
    # Sceglie se usare densità (normalizzata) o conteggi assoluti
    use_density = not use_counts
    h_temp, _, _ = np.histogram2d(x, y, bins=bins, density=use_density, range=range_limits)
    h_flat = h_temp.flatten()
    h_nonzero = h_flat[h_flat > 0]
    
    if len(h_nonzero) > 0:
        vmin = np.percentile(h_nonzero, vmin_percentile)
        vmax = np.percentile(h_nonzero, vmax_percentile)
        metric_name = "densità" if use_density else "conteggi"
        print(f"Limiti colormap ({metric_name}): vmin={vmin:.6f} ({vmin_percentile}%), vmax={vmax:.6f} ({vmax_percentile}%)")
        print(f"Range {metric_name} originale: [{h_nonzero.min():.6f}, {h_nonzero.max():.6f}]")
    else:
        vmin, vmax = None, None
        print("Nessun dato per calcolare i limiti colormap")
    
    # Crea la heatmap con diverse opzioni di visualizzazione
    if colormap == 'log':
        # Scala logaritmica per la colormap
        
        # Per log scale, usiamo solo vmax, vmin deve essere > 0
        safe_vmin = max(vmin, 1e-10) if vmin and vmin > 0 else 1e-10
        plt.hist2d(x, y, bins=bins, density=True, cmap='Blues', 
                  norm=LogNorm(vmin=safe_vmin, vmax=vmax), range=range_limits)
        plt.colorbar(label='Densità (log scale)')
    else:
        # Crea una colormap personalizzata che inizia da azzurro chiaro
        
        
        if colormap == 'Blues':
            # Crea una colormap Blues modificata: bianco per zero, poi azzurro chiaro -> blu scuro
            colors = [ 'lightblue', 'royalblue', 'blue', 'darkblue', 'navy']
            n_bins = 256
            custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_blues', colors, N=n_bins)
            # Imposta il valore zero come completamente bianco
            custom_cmap.set_bad('lightgray')
            custom_cmap.set_under('lightgray')
        else:
            # Per altre colormap, usa quelle standard
            custom_cmap = colormap
            
        # Per assicurarsi che i valori zero siano bianchi, impostiamo vmin leggermente sopra zero
        # se abbiamo valori zero nella heatmap
        h_for_zero_check, _, _ = np.histogram2d(x, y, bins=bins, density=use_density, range=range_limits)
        has_zeros = np.any(h_for_zero_check == 0)
        
        if has_zeros and vmin is not None and vmin <= 0:
            # Se abbiamo bin con valore zero e vmin è zero o negativo, impostiamo vmin a un valore piccolo
            min_nonzero = np.min(h_for_zero_check[h_for_zero_check > 0]) if np.any(h_for_zero_check > 0) else 1e-10
            adjusted_vmin = min_nonzero * 0.01  # Un valore molto piccolo ma > 0
            print(f"Valori zero rilevati, vmin aggiustato a: {adjusted_vmin:.10f}")
        else:
            adjusted_vmin = vmin
            
        colorbar_label = "Densità" if use_density else "Conteggi"
        plt.hist2d(x, y, bins=bins, density=use_density, cmap=custom_cmap, vmin=adjusted_vmin, vmax=vmax, range=range_limits)
        plt.colorbar(label=colorbar_label)
    
    # Imposta i limiti degli assi se specificati
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    # Crea la heatmap per analisi aggiuntiva
    h, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    print(f"Heatmap shape: {h.shape}")
    print(f"Valori heatmap min: {h.min():.6f}, max: {h.max():.6f}")
    print(f"Numero di bin non vuoti: {np.count_nonzero(h)}")
    print(f"Percentuale bin non vuoti: {np.count_nonzero(h)/(bins*bins)*100:.2f}%")
    
    # Analisi distribuzione dei valori nella heatmap
    h_flat = h.flatten()
    h_nonzero = h_flat[h_flat > 0]
    if len(h_nonzero) > 0:
        print(f"Densità media bin non vuoti: {h_nonzero.mean():.6f}")
        print(f"Densità mediana bin non vuoti: {np.median(h_nonzero):.6f}")
        print(f"Rapporto max/media: {h.max()/h_nonzero.mean():.2f}")
        print(f"Rapporto vmax/vmin: {vmax/vmin:.2f}" if vmin and vmax and vmin > 0 else "")
    
    # Identifica pattern a croce (se ha senso dopo preprocessing)
    if preprocessing in ['none', 'clip_percentile', 'winsorize']:
        center_x, center_y = x.mean(), y.mean()
        threshold = 0.05 * max(x.std(), y.std())  # Soglia adattiva
        near_x_axis = np.abs(y - center_y) < threshold
        near_y_axis = np.abs(x - center_x) < threshold
        
        cross_points = np.sum(near_x_axis) + np.sum(near_y_axis)
        cross_percentage = cross_points / len(x) * 100
        print(f"Punti su pattern a croce: {cross_points} ({cross_percentage:.2f}%)")
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{title} - {preprocessing} (colormap: {vmin_percentile}-{vmax_percentile}%)")
    plt.grid(True, alpha=0.3)
    
    return plt.gca()


def analyze_state_distribution(states):
    """Analizza la distribuzione degli stati."""
    print("\n=== ANALISI DISTRIBUZIONE STATI ===")
    print(f"Numero totale di stati: {len(states)}")
    print(f"Dimensioni stato: {states.shape[1]}")
    
    for i in range(min(states.shape[1], 4)):  # Analizza le prime 4 dimensioni
        dim_data = states[:, i]
        print(f"\nDimensione {i}:")
        print(f"  Min: {dim_data.min():.6f}")
        print(f"  Max: {dim_data.max():.6f}")
        print(f"  Media: {dim_data.mean():.6f}")
        print(f"  Std: {dim_data.std():.6f}")
        print(f"  Mediana: {np.median(dim_data):.6f}")
        
        # Analizza se ci sono valori concentrati
        unique_values = np.unique(dim_data)
        print(f"  Valori unici: {len(unique_values)}")
        if len(unique_values) <= 10:
            print(f"  Valori unici dettaglio: {unique_values}")
    
    # Controlla se ci sono pattern a forma di croce
    print("\n=== ANALISI PATTERN CROCE ===")
    if states.shape[1] >= 2:
        x, y = states[:, 0], states[:, 1]
        
        # Controlla concentrazione vicino agli assi
        center_x = x.mean()
        center_y = y.mean()
        print(f"Centro stimato: ({center_x:.6f}, {center_y:.6f})")
        
        # Diverse soglie per analizzare la distribuzione
        for threshold in [0.01, 0.02, 0.05, 0.1]:
            near_x_axis = np.abs(y - center_y) < threshold
            near_y_axis = np.abs(x - center_x) < threshold
            
            total_axis_points = np.sum(near_x_axis) + np.sum(near_y_axis)
            percentage = total_axis_points / len(states) * 100
            
            print(f"Soglia {threshold}: {total_axis_points} punti su assi ({percentage:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Genera heatmap degli stati dal dataset')
    parser.add_argument('--replay_dir', type=str, required=True,
                       help='Directory contenente i file del dataset (.npz)')
    parser.add_argument('--output_dir', type=str, default='./heatmaps',
                       help='Directory per salvare le heatmap generate')
    parser.add_argument('--bins', type=int, default=50,
                       help='Numero di bin per la heatmap')
    parser.add_argument('--show', action='store_true',
                       help='Mostra le plot invece di salvarle')
    parser.add_argument('--preprocessing', type=str, default='none',
                       choices=['none', 'clip_percentile', 'remove_outliers', 'winsorize', 'log_transform', 'robust_scale'],
                       help='Metodo di preprocessing per gestire outlier')
    parser.add_argument('--percentile', type=float, default=95,
                       help='Percentile per clipping/winsorizing (default: 95)')
    parser.add_argument('--std_factor', type=float, default=3,
                       help='Fattore sigma per rimozione outlier (default: 3)')
    parser.add_argument('--colormap', type=str, default='Blues',
                       choices=['Blues', 'viridis', 'plasma', 'hot', 'cool', 'log'],
                       help='Colormap per la heatmap')
    parser.add_argument('--vmin_percentile', type=float, default=0,
                       help='Percentile per valore minimo colormap (default: 0)')
    parser.add_argument('--vmax_percentile', type=float, default=95,
                       help='Percentile per valore massimo colormap (default: 95)')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, #[-0.3, 0.3],
                       help='Limiti asse X [min, max] (default: None)')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, #[-0.3, 0.3],
                       help='Limiti asse Y [min, max] (default: None)')
    parser.add_argument('--compare_methods', action='store_true',
                       help='Genera confronto tra diversi metodi di preprocessing')
    parser.add_argument('--use_counts', action='store_true',
                       help='Usa i conteggi numerici invece della densità per la colormap')
    
    args = parser.parse_args()
    
    # Crea directory di output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Caricamento dataset da: {args.replay_dir}")
    
    try:
        # Carica gli stati dal dataset
        states = collect_states_from_dataset(args.replay_dir)
        
        # Analizza la distribuzione
        analyze_state_distribution(states)
        
        # Crea heatmap singola
        print("\nGenerazione heatmap singola...")
        create_heatmap(states, "Distribuzione Stati - Point Mass Maze", 
                      bins=args.bins, preprocessing=args.preprocessing,
                      percentile=args.percentile, std_factor=args.std_factor,
                      colormap=args.colormap, vmin_percentile=args.vmin_percentile,
                      vmax_percentile=args.vmax_percentile, xlim=args.xlim, ylim=args.ylim,
                      use_counts=args.use_counts)
        
        if args.show:
            plt.show()
        else:
            filename = f'heatmap_{args.preprocessing}_{args.bins}bins.png'
            single_path = output_dir / filename
            plt.savefig(single_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap salvata in: {single_path}")
        
        plt.close()
        
        # Confronto tra metodi se richiesto
        if args.compare_methods:
            print("\nGenerazione confronto metodi...")
            methods = ['none', 'clip_percentile', 'remove_outliers', 'winsorize', 'robust_scale']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, method in enumerate(methods):
                if i >= len(axes):
                    break
                    
                plt.sca(axes[i])
                x, y, count = preprocess_states(states, method, args.percentile, args.std_factor)
                
                # Calcola limiti colormap
                h_temp, _, _ = np.histogram2d(x, y, bins=args.bins//2, density=True)
                h_flat = h_temp.flatten()
                h_nonzero = h_flat[h_flat > 0]
                
                if len(h_nonzero) > 0:
                    vmin = np.percentile(h_nonzero, args.vmin_percentile)
                    vmax = np.percentile(h_nonzero, args.vmax_percentile)
                else:
                    vmin, vmax = None, None
                
                # Usa colormap personalizzata per Blues
                if args.colormap == 'Blues':
                    colors = ['lightblue', 'royalblue', 'blue', 'darkblue', 'navy']
                    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_blues', colors, N=256)
                else:
                    custom_cmap = args.colormap
                
                plt.hist2d(x, y, bins=args.bins//2, density=True, cmap=custom_cmap, vmin=vmin, vmax=vmax)
                plt.title(f"{method} ({count} punti)")
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.colorbar()
            
            # Rimuovi subplot vuoti
            for i in range(len(methods), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            
            if args.show:
                plt.show()
            else:
                comp_path = output_dir / 'preprocessing_comparison.png'
                plt.savefig(comp_path, dpi=300, bbox_inches='tight')
                print(f"Confronto metodi salvato in: {comp_path}")
            
            plt.close()
        
        print("\nGenerazione completata!")
        
    except Exception as e:
        print(f"Errore: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())