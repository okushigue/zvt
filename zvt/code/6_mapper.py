#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_DATA_MAPPER.py - Mapping and Visualization of Resonance Data
Author: Jefferson M. Okushigue
Date: 2025-08-15
Updated to include data from all analysis scripts
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
import json
import glob
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Scientific configuration for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Constants for mapping - UPDATED WITH ALL CONSTANTS
ALL_CONSTANTS = {
    # Script 1 - Important Physical Constants
    'planck': 6.62607015e-34,
    'boltzmann': 1.380649e-23,
    'stefan_boltzmann': 5.670374419e-8,
    'wien': 2.897771955e-3,
    
    # Script 2 - Universal Constants
    'speed_of_light': 299792458.0,
    'gravitational': 6.67430e-11,
    'hbar': 1.054571817e-34,
    'mu_0': 1.25663706212e-6,
    
    # Script 3 - Fundamental Constants
    'fine_structure': 1/137.035999084,
    'electron_mass': 9.1093837015e-31,
    'rydberg': 1.0973731568160e7,
    'avogadro': 6.02214076e23,
    
    # Script 4 - Remaining Constants
    'gravity_acceleration': 9.80665,
    'gas_constant': 8.314462618,
    'vacuum_permittivity': 8.8541878128e-12,
    'elementary_charge': 1.602176634e-19,
    'proton_mass': 1.67262192369e-27,
    'bohr_radius': 5.29177210903e-11,
    'faraday_constant': 96485.33212,
    
    # Original Script - 4 Fundamental Forces
    'electromagnetic': 1 / 137.035999084,
    'strong': 0.1185,
    'weak': 0.0338,
    'gravitational_force': 5.906e-39,
    
    # Additional Constants for Analysis
    'weinberg_angle': 0.2312,
    'proton_electron': 1836.15267343,
    'euler_mascheroni': 0.5772156649,
    'fermi_coupling': 1.1663787e-5,
    'muon_electron': 206.7682826,
    'tau_electron': 3477.15,
    'neutron_proton': 1.00137841931,
    'dark_energy': 0.6847,
    'dark_matter': 0.2589,
    'baryon_density': 0.0486,
    'hubble_reduced': 0.6736,
    'sigma8': 0.8111,
    'gyromagnetic_proton': 2.7928473508,
    'gyromagnetic_neutron': 1.9130427,
    'magnetic_moment_ratio': 3.1524512605
}

# Physical categories for coloring - UPDATED
PHYSICS_CATEGORIES = {
    'Quantum Constants': ['planck', 'hbar', 'fine_structure', 'electron_mass', 'proton_mass', 'elementary_charge'],
    'Fundamental Forces': ['electromagnetic', 'strong', 'weak', 'gravitational_force'],
    'Thermodynamics': ['boltzmann', 'stefan_boltzmann', 'wien', 'gas_constant'],
    'Electromagnetism': ['speed_of_light', 'mu_0', 'vacuum_permittivity', 'faraday_constant'],
    'Gravitation': ['gravitational', 'gravity_acceleration'],
    'Atomic Structure': ['rydberg', 'bohr_radius'],
    'Chemistry/Macroscopic': ['avogadro'],
    'Cosmology': ['dark_energy', 'dark_matter', 'baryon_density', 'hubble_reduced', 'sigma8'],
    'Particle Physics': ['weinberg_angle', 'fermi_coupling', 'muon_electron', 'tau_electron', 'neutron_proton'],
    'Magnetic Properties': ['gyromagnetic_proton', 'gyromagnetic_neutron', 'magnetic_moment_ratio'],
    'Mathematical': ['euler_mascheroni', 'proton_electron']
}

# Colors by category - UPDATED
CATEGORY_COLORS = {
    'Quantum Constants': '#FF6B6B',
    'Fundamental Forces': '#4ECDC4', 
    'Thermodynamics': '#45B7D1',
    'Electromagnetism': '#96CEB4',
    'Gravitation': '#FECA57',
    'Atomic Structure': '#DDA0DD',
    'Chemistry/Macroscopic': '#98D8C8',
    'Cosmology': '#F7DC6F',
    'Particle Physics': '#BB8FCE',
    'Magnetic Properties': '#85C1E9',
    'Mathematical': '#F8C471'
}

class ZVTDataMapper:
    def __init__(self, cache_file="zeta_zeros_cache.pkl", results_dir="zvt_analysis_results"):
        self.cache_file = cache_file
        self.results_dir = results_dir
        self.maps_dir = os.path.join(results_dir, "comprehensive_maps")
        self.data = None
        self.resonances_df = None
        self.all_results = {}
        
        # Create maps directory
        os.makedirs(self.maps_dir, exist_ok=True)
        
        print("🗺️ ZVT DATA MAPPER - Comprehensive Scientific Visualization")
        print("=" * 70)
        
    def load_all_results(self):
        """Load results from all scripts"""
        print("📂 Loading data from all scripts...")
        
        # Load zeros from cache
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.zeros = pickle.load(f)
            print(f"✅ {len(self.zeros):,} zeros loaded from cache")
        else:
            print("❌ Cache not found!")
            return False
            
        # Load results from scripts
        self.load_script_results()
        
        # Build comprehensive DataFrame with all resonances
        self.build_comprehensive_dataframe()
        return True
    
    def load_script_results(self):
        """Load results from all executed scripts"""
        print("🔍 Searching for script results...")
        
        # Possible directories
        possible_dirs = [
            "zvt_constants_results",
            "zvt_constants_results_v2", 
            "zvt_universal_results",
            "zvt_fundamental_results",
            "zvt_remaining_results",
            "zvt_4forces_results"
        ]
        
        # Find report files
        report_files = []
        for dir_name in possible_dirs:
            if os.path.exists(dir_name):
                reports = glob.glob(f"{dir_name}/Relatorio_*.txt")
                report_files.extend(reports)
        
        print(f"📄 Found {len(report_files)} report files")
        
        # Extract data from reports
        for report_file in report_files:
            self.extract_data_from_report(report_file)
    
    def extract_data_from_report(self, report_file):
        """Extract resonance data from a report file"""
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Identify the script by file name
            script_name = os.path.basename(os.path.dirname(report_file))
            
            # Extract constants and values using regex
            import re
            
            # Pattern to find constants and their values
            pattern = r'(\w+(?:_\w+)*)\s*:\s*([\d\.eE+-]+)'
            matches = re.findall(pattern, content)
            
            for const_name, const_value in matches:
                try:
                    const_value = float(const_value)
                    if const_name in ALL_CONSTANTS:
                        # Find the best resonance for this constant
                        best_resonance = self.find_best_resonance(const_name, const_value)
                        if best_resonance:
                            best_resonance['source_script'] = script_name
                            self.all_results[const_name] = best_resonance
                except ValueError:
                    continue
                    
        except Exception as e:
            print(f"⚠️ Error processing {report_file}: {e}")
    
    def find_best_resonance(self, const_name, const_value):
        """Find the best resonance for a constant"""
        best_quality = float('inf')
        best_resonance = None
        
        # Search for best resonance for this constant
        for n, gamma in self.zeros:
            mod_val = gamma % const_value
            min_distance = min(mod_val, const_value - mod_val)
            
            if min_distance < best_quality:
                best_quality = min_distance
                best_resonance = {
                    'constant': const_name,
                    'constant_value': const_value,
                    'zero_index': n,
                    'gamma': gamma,
                    'quality': min_distance,
                    'error_percent': (min_distance / const_value) * 100,
                    'energy_gev': gamma / 10,
                    'log_quality': np.log10(min_distance),
                    'log_constant': np.log10(const_value),
                    'category': self.get_category(const_name)
                }
        
        return best_resonance
    
    def build_comprehensive_dataframe(self):
        """Build comprehensive DataFrame with all found resonances"""
        print("🔬 Building comprehensive DataFrame...")
        
        resonances_data = []
        
        # Add found results
        for const_name, resonance_data in self.all_results.items():
            resonances_data.append(resonance_data)
        
        # For constants not found in reports, calculate
        for const_name, const_value in ALL_CONSTANTS.items():
            if const_name not in self.all_results:
                best_resonance = self.find_best_resonance(const_name, const_value)
                if best_resonance:
                    best_resonance['source_script'] = 'calculated'
                    resonances_data.append(best_resonance)
        
        self.resonances_df = pd.DataFrame(resonances_data)
        print(f"✅ DataFrame built with {len(self.resonances_df)} resonances")
        
        # Basic statistics
        found_count = len([r for r in resonances_data if r.get('source_script') != 'calculated'])
        print(f"📊 {found_count} resonances found in reports")
        print(f"📊 {len(resonances_data) - found_count} calculated now")
    
    def get_category(self, const_name):
        """Return physical category of the constant"""
        for category, constants in PHYSICS_CATEGORIES.items():
            if const_name in constants:
                return category
        return 'Others'
    
    def map_comprehensive_overview(self):
        """General comprehensive map of all constants"""
        print("🌍 Generating comprehensive overview map...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Graph 1: All constants by quality
        categories = list(PHYSICS_CATEGORIES.keys())
        for category in categories:
            cat_data = self.resonances_df[self.resonances_df['category'] == category]
            if not cat_data.empty:
                color = CATEGORY_COLORS.get(category, 'gray')
                marker = 'o' if cat_data.iloc[0].get('source_script') != 'calculated' else '^'
                ax1.scatter(cat_data['energy_gev'], cat_data['quality'], 
                           c=color, s=100, alpha=0.7, label=category, 
                           edgecolors='black', marker=marker)
        
        ax1.set_xlabel('Energy (GeV)', fontsize=12)
        ax1.set_ylabel('Resonance Quality', fontsize=12)
        ax1.set_title('Comprehensive Map: Energy vs Quality', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Graph 2: Distribution by category
        category_counts = self.resonances_df['category'].value_counts()
        colors = [CATEGORY_COLORS.get(cat, 'gray') for cat in category_counts.index]
        
        wedges, texts, autotexts = ax2.pie(category_counts.values, labels=category_counts.index, 
                                            colors=colors, autopct='%1.0f%%', startangle=90)
        ax2.set_title('Distribution by Physical Category', fontsize=14, fontweight='bold')
        
        # Graph 3: Top 15 best resonances
        top_15 = self.resonances_df.nsmallest(15, 'quality')
        colors = [CATEGORY_COLORS.get(cat, 'gray') for cat in top_15['category']]
        
        bars = ax3.barh(range(len(top_15)), -top_15['log_quality'], color=colors, alpha=0.8, edgecolor='black')
        ax3.set_yticks(range(len(top_15)))
        ax3.set_yticklabels([f"{const[:12]}..." for const in top_15['constant']], fontsize=8)
        ax3.set_xlabel('Log₁₀(Quality) - Better →', fontsize=12)
        ax3.set_title('Top 15 Best Resonances', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Add error values
        for i, (_, row) in enumerate(top_15.iterrows()):
            ax3.text(-row['log_quality'] + 0.1, i, f"{row['error_percent']:.1e}%", 
                   va='center', fontsize=7, fontweight='bold')
        
        # Graph 4: Comparison between scripts
        script_counts = self.resonances_df['source_script'].value_counts()
        ax4.bar(range(len(script_counts)), script_counts.values, color='skyblue', alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(script_counts)))
        ax4.set_xticklabels([s.replace('_', ' ').title() for s in script_counts.index], rotation=45, ha='right')
        ax4.set_ylabel('Number of Resonances')
        ax4.set_title('Resonances by Script', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "comprehensive_overview_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: {filename}")
    
    def map_quality_comparison(self):
        """Comparative quality map between categories"""
        print("⚖️ Generating comparative quality map...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Box plot by category
        categories = list(PHYSICS_CATEGORIES.keys())
        qualities_by_cat = []
        cat_labels = []
        
        for category in categories:
            cat_data = self.resonances_df[self.resonances_df['category'] == category]
            if not cat_data.empty:
                qualities_by_cat.append(cat_data['log_quality'].values)
                cat_labels.append(category)
        
        bp = ax1.boxplot(qualities_by_cat, labels=cat_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], [CATEGORY_COLORS.get(cat, 'gray') for cat in cat_labels]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Log₁₀(Quality)')
        ax1.set_title('Quality Distribution by Category', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Violin plot by category
        parts = ax2.violinplot(qualities_by_cat, positions=range(len(cat_labels)))
        
        for i, (pc, color) in enumerate(zip(parts['bodies'], [CATEGORY_COLORS.get(cat, 'gray') for cat in cat_labels])):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax2.set_xticks(range(len(cat_labels)))
        ax2.set_xticklabels(cat_labels, rotation=45)
        ax2.set_ylabel('Log₁₀(Quality)')
        ax2.set_title('Quality Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "quality_comparison_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: {filename}")
    
    def map_energy_spectrum(self):
        """Energy spectrum map"""
        print("⚡ Generating energy spectrum map...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Energy histogram
        energies = self.resonances_df['energy_gev']
        ax1.hist(energies, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(energies.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {energies.mean():.0f} GeV')
        ax1.axvline(energies.median(), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {energies.median():.0f} GeV')
        ax1.set_xlabel('Energy (GeV)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Energy Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Energy vs Quality by category
        for category in PHYSICS_CATEGORIES.keys():
            cat_data = self.resonances_df[self.resonances_df['category'] == category]
            if not cat_data.empty:
                ax2.scatter(cat_data['energy_gev'], cat_data['quality'], 
                           c=CATEGORY_COLORS.get(category, 'gray'), 
                           s=100, alpha=0.7, label=category, edgecolors='black')
        
        ax2.set_xlabel('Energy (GeV)')
        ax2.set_ylabel('Quality')
        ax2.set_yscale('log')
        ax2.set_title('Energy vs Quality by Category', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Heatmap: Category vs Energy Range
        energy_bins = [0, 10000, 50000, 100000, 200000]
        energy_labels = ['0-10 TeV', '10-50 TeV', '50-100 TeV', '>100 TeV']
        
        heatmap_data = []
        for category in PHYSICS_CATEGORIES.keys():
            cat_data = self.resonances_df[self.resonances_df['category'] == category]
            row = []
            for i in range(len(energy_bins)-1):
                count = len(cat_data[(cat_data['energy_gev'] >= energy_bins[i]) & 
                                   (cat_data['energy_gev'] < energy_bins[i+1])])
                row.append(count)
            heatmap_data.append(row)
        
        im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(energy_labels)))
        ax3.set_xticklabels(energy_labels, rotation=45)
        ax3.set_yticks(range(len(PHYSICS_CATEGORIES.keys())))
        ax3.set_yticklabels([cat[:15] for cat in PHYSICS_CATEGORIES.keys()])
        ax3.set_title('Heatmap: Category vs Energy Range', fontsize=14, fontweight='bold')
        
        # Add values in cells
        for i in range(len(PHYSICS_CATEGORIES.keys())):
            for j in range(len(energy_labels)):
                ax3.text(j, i, str(heatmap_data[i][j]), ha='center', va='center', 
                         color='white' if heatmap_data[i][j] > 5 else 'black')
        
        plt.colorbar(im, ax=ax3)
        
        # Quality evolution with energy
        sorted_df = self.resonances_df.sort_values('energy_gev')
        ax4.plot(sorted_df['energy_gev'], sorted_df['quality'], 'o-', alpha=0.7, color='purple')
        ax4.fill_between(sorted_df['energy_gev'], sorted_df['quality'], alpha=0.3, color='purple')
        ax4.set_xlabel('Energy (GeV)')
        ax4.set_ylabel('Quality')
        ax4.set_yscale('log')
        ax4.set_title('Quality Evolution with Energy', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "energy_spectrum_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: {filename}")
    
    def map_physics_relations_network(self):
        """Physical relations network map"""
        print("🌐 Generating physical relations network...")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (constants)
        for _, row in self.resonances_df.iterrows():
            G.add_node(row['constant'], 
                      category=row['category'],
                      quality=row['quality'],
                      energy=row['energy_gev'],
                      script=row.get('source_script', 'unknown'))
        
        # Add edges based on energy and category similarity
        for i, row1 in self.resonances_df.iterrows():
            for j, row2 in self.resonances_df.iterrows():
                if i < j:  # Avoid duplicates
                    # Connect if same category or similar energy
                    same_category = row1['category'] == row2['category']
                    energy_diff = abs(row1['energy_gev'] - row2['energy_gev'])
                    similar_energy = energy_diff < 20000  # 20 TeV
                    
                    if same_category or similar_energy:
                        weight = 1 / (energy_diff + 1)  # Weight inversely proportional to distance
                        if same_category:
                            weight *= 2  # Higher weight for same category
                        G.add_edge(row1['constant'], row2['constant'], weight=weight)
        
        # Graph layout with spring layout
        pos = nx.spring_layout(G, k=2, iterations=100)
        
        # Draw nodes by category
        for category, color in CATEGORY_COLORS.items():
            nodes_in_cat = [node for node in G.nodes() 
                           if G.nodes[node]['category'] == category]
            if nodes_in_cat:
                # Node size based on quality (lower quality = larger node)
                node_sizes = []
                for node in nodes_in_cat:
                    quality = G.nodes[node]['quality']
                    # Normalize size between 300 and 1500
                    size = 300 + 1200 * (1 - min(quality, 1e-10) / 1e-10)
                    node_sizes.append(size)
                
                nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_cat, 
                                     node_color=color, node_size=node_sizes, 
                                     alpha=0.8, ax=ax)
        
        # Draw edges with thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] * 2 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.3, edge_color='gray', ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Customize
        ax.set_title('Network of Physical Constants Relations\n' + 
                     '(Connected by Category or Energy Proximity)', 
                     fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Legend
        legend_elements = [plt.scatter([], [], c=color, s=100, alpha=0.8, edgecolors='black') 
                           for color in CATEGORY_COLORS.values()]
        ax.legend(legend_elements, CATEGORY_COLORS.keys(), 
                 loc='upper left', fontsize=10)
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "physics_relations_network_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: {filename}")
    
    def map_clustering_and_patterns(self):
        """Clustering and pattern analysis"""
        print("🔍 Generating clustering and pattern analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Prepare data for clustering
        features = ['log_constant', 'log_quality', 'energy_gev']
        X = self.resonances_df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means Clustering
        n_clusters = min(5, len(X_scaled))  # Cannot have more clusters than samples
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters_kmeans = kmeans.fit_predict(X_scaled)
        
        scatter = ax1.scatter(self.resonances_df['energy_gev'], self.resonances_df['quality'], 
                             c=clusters_kmeans, cmap='viridis', s=100, alpha=0.8, edgecolors='black')
        ax1.set_xlabel('Energy (GeV)')
        ax1.set_ylabel('Quality')
        ax1.set_yscale('log')
        ax1.set_title(f'K-Means Clustering (k={n_clusters})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add centroids
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        ax1.scatter(centers[:, 2], 10**centers[:, 1], c='red', s=200, marker='X', 
                   edgecolors='black', linewidth=2, label='Centroids')
        ax1.legend()
        
        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        clusters_dbscan = dbscan.fit_predict(X_scaled)
        
        ax2.scatter(self.resonances_df['energy_gev'], self.resonances_df['quality'], 
                    c=clusters_dbscan, cmap='plasma', s=100, alpha=0.8, edgecolors='black')
        ax2.set_xlabel('Energy (GeV)')
        ax2.set_ylabel('Quality')
        ax2.set_yscale('log')
        ax2.set_title('DBSCAN Clustering', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # t-SNE Visualization
        if len(X_scaled) >= 4:
            perplexity = min(5, len(X_scaled) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            X_tsne = tsne.fit_transform(X_scaled)
            
            colors = [CATEGORY_COLORS.get(cat, 'gray') for cat in self.resonances_df['category']]
            ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=100, alpha=0.8, edgecolors='black')
            ax3.set_xlabel('t-SNE Dimension 1')
            ax3.set_ylabel('t-SNE Dimension 2')
            ax3.set_title(f't-SNE: Dimensionality Reduction (perplexity={perplexity})', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add labels to identify important points
            top_5 = self.resonances_df.nsmallest(5, 'quality')
            for i, (_, row) in enumerate(top_5.iterrows()):
                idx = self.resonances_df.index.get_loc(row.name)
                ax3.annotate(row['constant'][:8], (X_tsne[idx, 0], X_tsne[idx, 1]), 
                           xytext=(3, 3), textcoords='offset points', fontsize=6, alpha=0.7)
        else:
            ax3.text(0.5, 0.5, 'Insufficient samples\nfor t-SNE', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('t-SNE: Dimensionality Reduction', fontsize=14, fontweight='bold')
        
        # Correlation matrix
        corr_features = ['log_constant', 'log_quality', 'energy_gev', 'error_percent']
        corr_matrix = self.resonances_df[corr_features].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax4, fmt='.2f')
        ax4.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "clustering_patterns_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: {filename}")
    
    def map_script_comparison(self):
        """Comparative map between scripts"""
        print("📊 Generating comparative map between scripts...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Script performance
        script_stats = self.resonances_df.groupby('source_script').agg({
            'quality': ['count', 'mean', 'min'],
            'energy_gev': 'mean'
        }).round(2)
        
        # Number of resonances by script
        script_counts = self.resonances_df['source_script'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(script_counts)))
        
        bars = ax1.bar(range(len(script_counts)), script_counts.values, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xticks(range(len(script_counts)))
        ax1.set_xticklabels([s.replace('_', ' ').title() for s in script_counts.index], rotation=45, ha='right')
        ax1.set_ylabel('Number of Resonances')
        ax1.set_title('Resonances Found by Script', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Average quality by script
        quality_by_script = self.resonances_df.groupby('source_script')['quality'].mean().sort_values()
        ax2.barh(range(len(quality_by_script)), -np.log10(quality_by_script.values), 
                color=colors, alpha=0.8, edgecolor='black')
        ax2.set_yticks(range(len(quality_by_script)))
        ax2.set_yticklabels([s.replace('_', ' ').title() for s in quality_by_script.index])
        ax2.set_xlabel('-Log₁₀(Average Quality)')
        ax2.set_title('Average Quality by Script', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Average energy by script
        energy_by_script = self.resonances_df.groupby('source_script')['energy_gev'].mean().sort_values()
        ax3.barh(range(len(energy_by_script)), energy_by_script.values, 
                color=colors, alpha=0.8, edgecolor='black')
        ax3.set_yticks(range(len(energy_by_script)))
        ax3.set_yticklabels([s.replace('_', ' ').title() for s in energy_by_script.index])
        ax3.set_xlabel('Average Energy (GeV)')
        ax3.set_title('Average Energy by Script', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Best resonance by script
        best_by_script = self.resonances_df.loc[self.resonances_df.groupby('source_script')['quality'].idxmin()]
        ax4.scatter(best_by_script['source_script'], -np.log10(best_by_script['quality']), 
                   s=200, c=[CATEGORY_COLORS.get(cat, 'gray') for cat in best_by_script['category']], 
                   alpha=0.8, edgecolors='black')
        
        # Add labels
        for i, (_, row) in enumerate(best_by_script.iterrows()):
            ax4.annotate(f"{row['constant'][:10]}...", 
                        (row['source_script'], -np.log10(row['quality'])),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Script')
        ax4.set_ylabel('-Log₁₀(Quality)')
        ax4.set_title('Best Resonance by Script', fontsize=14, fontweight='bold')
        ax4.set_xticklabels([s.replace('_', ' ').title() for s in best_by_script['source_script'].unique()], 
                           rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "script_comparison_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: {filename}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive mapping report"""
        print("📋 Generating comprehensive mapping report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.maps_dir, f"Comprehensive_Mapping_Report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ZVT DATA MAPPER - COMPREHENSIVE MAPPING REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Zeros analyzed: {len(self.zeros):,}\n")
            f.write(f"Constants mapped: {len(self.resonances_df)}\n")
            f.write(f"Physical categories: {len(PHYSICS_CATEGORIES)}\n\n")
            
            f.write("GENERAL STATISTICS:\n")
            f.write(f"Best overall quality: {self.resonances_df['quality'].min():.2e}\n")
            f.write(f"Worst overall quality: {self.resonances_df['quality'].max():.2e}\n")
            f.write(f"Average energy: {self.resonances_df['energy_gev'].mean():.2f} GeV\n")
            f.write(f"Median energy: {self.resonances_df['energy_gev'].median():.2f} GeV\n")
            f.write(f"Energy standard deviation: {self.resonances_df['energy_gev'].std():.2f} GeV\n\n")
            
            f.write("TOP 10 BEST RESONANCES:\n")
            top_10 = self.resonances_df.nsmallest(10, 'quality')
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"{i:2d}. {row['constant']:<25} | Error: {row['error_percent']:.2e}%\n")
                f.write(f"     Zero #{row['zero_index']:>10,} | γ={row['gamma']:.6f} | E={row['energy_gev']:.2f} GeV\n")
                f.write(f"     Category: {row['category']:<20} | Script: {row.get('source_script', 'N/A')}\n")
            
            f.write(f"\nBY CATEGORY:\n")
            for category in PHYSICS_CATEGORIES.keys():
                cat_data = self.resonances_df[self.resonances_df['category'] == category]
                if not cat_data.empty:
                    best_in_cat = cat_data.loc[cat_data['quality'].idxmin()]
                    f.write(f"{category}:\n")
                    f.write(f"  • Total: {len(cat_data)} constants\n")
                    f.write(f"  • Best: {best_in_cat['constant']} ({best_in_cat['error_percent']:.2e}%)\n")
                    f.write(f"  • Average energy: {cat_data['energy_gev'].mean():.2f} GeV\n")
            
            f.write(f"\nBY SCRIPT:\n")
            script_stats = self.resonances_df.groupby('source_script').agg({
                'quality': ['count', 'min', 'mean'],
                'energy_gev': ['mean', 'std']
            }).round(2)
            
            for script in script_stats.index:
                stats = script_stats.loc[script]
                f.write(f"{script}:\n")
                f.write(f"  • Resonances: {stats[('quality', 'count')]}\n")
                f.write(f"  • Best quality: {stats[('quality', 'min')]:.2e}\n")
                f.write(f"  • Average quality: {stats[('quality', 'mean')]:.2e}\n")
                f.write(f"  • Average energy: {stats[('energy_gev', 'mean')]:.2f} ± {stats[('energy_gev', 'std')]:.2f} GeV\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("- comprehensive_overview_map.png\n")
            f.write("- quality_comparison_map.png\n")  
            f.write("- energy_spectrum_map.png\n")
            f.write("- physics_relations_network_map.png\n")
            f.write("- clustering_patterns_map.png\n")
            f.write("- script_comparison_map.png\n")
            f.write("="*80 + "\n")
        
        print(f"📊 Report saved: {report_file}")
    
    def run_all_mappings(self):
        """Execute all mappings"""
        print("\n🚀 STARTING COMPREHENSIVE MAPPING...")
        print("="*70)
        
        if not self.load_all_results():
            return
            
        print(f"\n📊 Dataset loaded:")
        print(f"   • {len(self.zeros):,} zeta zeros")
        print(f"   • {len(self.resonances_df)} constants analyzed")
        print(f"   • {len(PHYSICS_CATEGORIES)} physical categories")
        print(f"   • {len(self.resonances_df['source_script'].unique())} scripts analyzed")
        
        print(f"\n🗺️ Generating comprehensive maps...")
        
        # Execute all mappings
        self.map_comprehensive_overview()
        self.map_quality_comparison()
        self.map_energy_spectrum()
        self.map_physics_relations_network()
        self.map_clustering_and_patterns()
        self.map_script_comparison()
        
        # Generate report
        self.generate_comprehensive_report()
        
        print(f"\n✅ MAPPING COMPLETE!")
        print(f"📁 All maps saved in: {self.maps_dir}")
        print("="*70)

def main():
    """Main function"""
    mapper = ZVTDataMapper()
    mapper.run_all_mappings()

if __name__ == "__main__":
    main()
