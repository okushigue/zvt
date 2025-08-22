#!/usr/bin/env python3
"""
zvt_neutrino_data_analyzer.py - Analyzes real neutrino data using ZVT theory
Tool for collaborations like IceCube, ANTARES, KM3NeT

REVIEWED VERSION with improvements and fixes
"""
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZVTNeutrinoDataAnalyzer:
    """Neutrino data analyzer using ZVT theory"""
    
    def __init__(self, config: Dict[str, float] = None):
        """
        Initialize the analyzer with configurable parameters.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.zvt_parameters = {
            'energy_scaling': 1e-25,  # Energy scaling factor
            'phase_threshold': 0.1,   # Threshold for resonance
            'detection_threshold': 0.3  # Detection probability threshold
        }
        
        # Update parameters with user config if provided
        if config:
            self.zvt_parameters.update(config)
            
        logger.info(f"ZVT Analyzer initialized with parameters: {self.zvt_parameters}")
    
    def load_icecube_format_data(self, data_file: str = None, n_events: int = 1000) -> List[Dict[str, Any]]:
        """
        Loads data in IceCube format (simulated for demo purposes).
        
        Args:
            data_file: Path to data file (not used in simulation)
            n_events: Number of events to simulate
            
        Returns:
            List of event dictionaries
        """
        logger.info(f"Generating {n_events} simulated neutrino events")
        
        events = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_events):
            try:
                # Energy following power law distribution (E^-2) - typical for cosmic rays
                log_energy = np.random.uniform(12, 18)  # log10(E/eV) from 12 to 18
                energy = 10**log_energy
                
                # Direction (zenith: 0=down, pi=up; azimuth: 0 to 2pi)
                zenith = np.random.uniform(0, math.pi)
                azimuth = np.random.uniform(0, 2*math.pi)
                
                # Random time within a year
                base_time = datetime(2024, 1, 1)
                event_time = base_time + timedelta(seconds=np.random.uniform(0, 365*24*3600))
                
                # Event quality based on typical detector performance
                quality_weights = [0.7, 0.2, 0.1]  # good, poor, excellent
                quality = np.random.choice(['good', 'poor', 'excellent'], p=quality_weights)
                
                event = {
                    'event_id': f"IC{2024}{i:06d}",
                    'energy_eV': energy,
                    'log_energy': log_energy,  # Added for convenience
                    'zenith': zenith,
                    'azimuth': azimuth,
                    'time': event_time.isoformat(),
                    'quality': quality,
                    'detector_string': np.random.randint(1, 87),  # IceCube has 86 strings
                    'reconstruction_likelihood': np.random.uniform(0.1, 1.0)
                }
                
                events.append(event)
                
            except Exception as e:
                logger.error(f"Error generating event {i}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(events)} events")
        return events
    
    def calculate_zvt_signature(self, event: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates ZVT signature for a neutrino event.
        
        Args:
            event: Event dictionary containing energy and other parameters
            
        Returns:
            Dictionary with ZVT signature parameters
        """
        try:
            energy = event['energy_eV']
            
            # Physical constants
            hbar = 1.054571817e-34  # Reduced Planck constant (J⋅s)
            energy_joules = energy * 1.602176634e-19  # Convert eV to Joules
            
            # Calculate Compton frequency
            compton_frequency = energy_joules / hbar
            
            # ZVT quantum angle calculation
            quantum_angle = (compton_frequency * self.zvt_parameters['energy_scaling']) % (2 * math.pi)
            
            # Corresponding Riemann zero (theoretical connection)
            riemann_zero = compton_frequency * self.zvt_parameters['energy_scaling'] / (2 * math.pi)
            
            # ZVT strength based on resonance condition
            zvt_strength = abs(math.cos(quantum_angle))
            
            # Additional ZVT parameters
            phase_coherence = math.exp(-abs(quantum_angle - math.pi) / math.pi)
            resonance_factor = 1.0 / (1.0 + abs(quantum_angle - math.pi/2)**2)
            
            return {
                'quantum_angle': quantum_angle,
                'riemann_zero': riemann_zero,
                'zvt_strength': zvt_strength,
                'phase_coherence': phase_coherence,
                'resonance_factor': resonance_factor,
                'compton_frequency': compton_frequency
            }
            
        except Exception as e:
            logger.error(f"Error calculating ZVT signature for event {event.get('event_id', 'unknown')}: {e}")
            return {
                'quantum_angle': 0.0,
                'riemann_zero': 0.0,
                'zvt_strength': 0.0,
                'phase_coherence': 0.0,
                'resonance_factor': 0.0,
                'compton_frequency': 0.0
            }
    
    def classify_event_source(self, event: Dict[str, Any]) -> Dict[str, str]:
        """
        Classifies astrophysical source using ZVT analysis.
        
        Args:
            event: Event dictionary
            
        Returns:
            Classification results
        """
        try:
            zvt_sig = self.calculate_zvt_signature(event)
            energy = event['energy_eV']
            zenith = event['zenith']
            
            # Enhanced classification logic
            if energy > 1e16:  # > 10 PeV - Very High Energy
                if zvt_sig['zvt_strength'] > 0.8 and zvt_sig['phase_coherence'] > 0.6:
                    source_type = 'Extragalactic'
                    confidence = 'High'
                    if zenith > math.pi/2:  # Upgoing (through Earth)
                        astrophysical_type = 'AGN or Blazar'
                    else:
                        astrophysical_type = 'GRB or Transient'
                else:
                    source_type = 'Cosmic Ray Induced'
                    confidence = 'Medium'
                    astrophysical_type = 'Atmospheric'
                    
            elif energy > 1e14:  # > 100 TeV - High Energy
                if zvt_sig['zvt_strength'] > 0.7:
                    source_type = 'Galactic'
                    confidence = 'High'
                    astrophysical_type = 'Supernova Remnant or Pulsar'
                elif zvt_sig['resonance_factor'] > 0.5:
                    source_type = 'Galactic Diffuse'
                    confidence = 'Medium'
                    astrophysical_type = 'Cosmic Ray Interaction'
                else:
                    source_type = 'Atmospheric'
                    confidence = 'Medium'
                    astrophysical_type = 'Conventional'
                    
            else:  # < 100 TeV - Moderate Energy
                if zvt_sig['zvt_strength'] > 0.6:
                    source_type = 'Solar/Atmospheric'
                    confidence = 'Low'
                    astrophysical_type = 'Solar Neutrino or Atmospheric'
                else:
                    source_type = 'Background'
                    confidence = 'Low'
                    astrophysical_type = 'Detector Noise'
            
            return {
                'source': source_type,
                'confidence': confidence,
                'type': astrophysical_type,
                'zvt_score': zvt_sig['zvt_strength']
            }
            
        except Exception as e:
            logger.error(f"Error classifying event {event.get('event_id', 'unknown')}: {e}")
            return {
                'source': 'Unknown',
                'confidence': 'Low',
                'type': 'Classification Error',
                'zvt_score': 0.0
            }
    
    def find_zvt_clusters(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Finds event clusters using ZVT phase patterns.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of clusters (each cluster is a list of events with ZVT signatures)
        """
        logger.info(f"Searching for ZVT clusters in {len(events)} events")
        
        clusters = []
        zvt_signatures = []
        
        # Calculate ZVT signatures for all events
        for event in events:
            try:
                zvt_sig = self.calculate_zvt_signature(event)
                if zvt_sig['zvt_strength'] > 0.1:  # Only consider significant events
                    zvt_signatures.append({
                        'event': event,
                        'quantum_angle': zvt_sig['quantum_angle'],
                        'riemann_zero': zvt_sig['riemann_zero'],
                        'zvt_strength': zvt_sig['zvt_strength'],
                        'phase_coherence': zvt_sig['phase_coherence']
                    })
            except Exception as e:
                logger.error(f"Error processing event for clustering: {e}")
                continue
        
        if not zvt_signatures:
            logger.warning("No events with significant ZVT signatures found")
            return clusters
        
        # Clustering algorithm based on quantum angle similarity
        angle_threshold = self.zvt_parameters['phase_threshold']
        used = [False] * len(zvt_signatures)
        
        for i, sig1 in enumerate(zvt_signatures):
            if used[i]:
                continue
                
            cluster = [sig1]
            used[i] = True
            
            # Find similar events
            for j in range(i + 1, len(zvt_signatures)):
                if used[j]:
                    continue
                    
                sig2 = zvt_signatures[j]
                
                # Check quantum angle similarity (considering circular nature)
                angle_diff = abs(sig1['quantum_angle'] - sig2['quantum_angle'])
                angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                
                # Also check phase coherence similarity
                coherence_diff = abs(sig1['phase_coherence'] - sig2['phase_coherence'])
                
                if (angle_diff < angle_threshold and coherence_diff < 0.2):
                    cluster.append(sig2)
                    used[j] = True
            
            # Only keep significant clusters
            if len(cluster) >= 3:  # Minimum cluster size
                cluster_strength = np.mean([sig['zvt_strength'] for sig in cluster])
                if cluster_strength > 0.5:  # Minimum cluster strength
                    clusters.append(cluster)
        
        logger.info(f"Found {len(clusters)} significant ZVT clusters")
        return clusters
    
    def analyze_zvt_enhanced_sensitivity(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyzes sensitivity improvement with ZVT optimization.
        
        Args:
            events: List of events to analyze
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        logger.info("Analyzing ZVT-enhanced sensitivity")
        
        traditional_detections = 0
        zvt_detections = 0
        high_confidence_zvt = 0
        
        energy_threshold_traditional = 1e14  # 100 TeV traditional threshold
        
        for event in events:
            try:
                zvt_sig = self.calculate_zvt_signature(event)
                
                # Traditional detection (energy-only threshold)
                if event['energy_eV'] > energy_threshold_traditional:
                    traditional_detections += 1
                
                # ZVT-enhanced detection (resonance-based)
                if zvt_sig['zvt_strength'] > self.zvt_parameters['detection_threshold']:
                    zvt_detections += 1
                    
                    # High confidence ZVT detections
                    if (zvt_sig['zvt_strength'] > 0.7 and 
                        zvt_sig['phase_coherence'] > 0.5):
                        high_confidence_zvt += 1
                        
            except Exception as e:
                logger.error(f"Error in sensitivity analysis for event: {e}")
                continue
        
        # Calculate metrics
        sensitivity_gain = zvt_detections / max(traditional_detections, 1)
        high_confidence_ratio = high_confidence_zvt / max(zvt_detections, 1)
        
        results = {
            'traditional_detections': traditional_detections,
            'zvt_detections': zvt_detections,
            'high_confidence_zvt': high_confidence_zvt,
            'sensitivity_gain': sensitivity_gain,
            'improvement_percent': (sensitivity_gain - 1) * 100,
            'high_confidence_ratio': high_confidence_ratio,
            'total_events': len(events)
        }
        
        logger.info(f"Sensitivity analysis complete: {results}")
        return results
    
    def generate_zvt_alerts(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates alerts for high significance ZVT events.
        
        Args:
            events: List of events to analyze
            
        Returns:
            List of alert dictionaries
        """
        logger.info("Generating ZVT alerts")
        
        alerts = []
        
        for event in events:
            try:
                zvt_sig = self.calculate_zvt_signature(event)
                classification = self.classify_event_source(event)
                
                # Alert criteria
                is_high_energy = event['energy_eV'] > 1e15  # > 1 PeV
                is_strong_zvt = zvt_sig['zvt_strength'] > 0.8
                is_high_confidence = classification['confidence'] == 'High'
                is_coherent = zvt_sig['phase_coherence'] > 0.6
                
                if (is_high_energy and is_strong_zvt and is_high_confidence):
                    # Determine alert priority
                    if zvt_sig['zvt_strength'] > 0.9 and is_coherent:
                        priority = 'Critical'
                        recommended_action = 'Immediate multi-messenger follow-up'
                    elif zvt_sig['zvt_strength'] > 0.85:
                        priority = 'High'
                        recommended_action = 'Rapid follow-up within 24h'
                    else:
                        priority = 'Medium'
                        recommended_action = 'Standard follow-up procedure'
                    
                    alert = {
                        'event_id': event['event_id'],
                        'timestamp': event['time'],
                        'energy_eV': event['energy_eV'],
                        'energy_log10': math.log10(event['energy_eV']),
                        'zenith': event['zenith'],
                        'azimuth': event['azimuth'],
                        'zvt_strength': zvt_sig['zvt_strength'],
                        'phase_coherence': zvt_sig['phase_coherence'],
                        'predicted_source': classification['source'],
                        'source_type': classification['type'],
                        'alert_type': 'High-Energy Astrophysical Neutrino',
                        'priority': priority,
                        'confidence': classification['confidence'],
                        'recommended_action': recommended_action,
                        'coordinates': {
                            'zenith_deg': math.degrees(event['zenith']),
                            'azimuth_deg': math.degrees(event['azimuth'])
                        }
                    }
                    alerts.append(alert)
                    
            except Exception as e:
                logger.error(f"Error generating alert for event {event.get('event_id', 'unknown')}: {e}")
                continue
        
        # Sort alerts by priority and ZVT strength
        priority_order = {'Critical': 3, 'High': 2, 'Medium': 1}
        alerts.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['zvt_strength']), reverse=True)
        
        logger.info(f"Generated {len(alerts)} ZVT alerts")
        return alerts
    
    def export_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Export analysis results to JSON file.
        
        Args:
            results: Results dictionary to export
            filename: Output filename (optional)
            
        Returns:
            Filename of exported results
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zvt_analysis_results_{timestamp}.json"
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Deep convert the results
            json_results = json.loads(json.dumps(results, default=convert_numpy))
            
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
                
            logger.info(f"Results exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return None

def main():
    """Example usage of the ZVT Neutrino Data Analyzer"""
    
    # Initialize analyzer
    analyzer = ZVTNeutrinoDataAnalyzer()
    
    # Load simulated data
    events = analyzer.load_icecube_format_data(n_events=500)
    
    # Perform ZVT analysis
    sensitivity_results = analyzer.analyze_zvt_enhanced_sensitivity(events)
    clusters = analyzer.find_zvt_clusters(events)
    alerts = analyzer.generate_zvt_alerts(events)
    
    # Compile results
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_events': len(events),
        'sensitivity_analysis': sensitivity_results,
        'zvt_clusters': len(clusters),
        'alerts_generated': len(alerts),
        'high_priority_alerts': len([a for a in alerts if a['priority'] in ['Critical', 'High']]),
        'cluster_details': [
            {
                'cluster_id': i,
                'size': len(cluster),
                'mean_zvt_strength': np.mean([sig['zvt_strength'] for sig in cluster]),
                'mean_energy': np.mean([sig['event']['energy_eV'] for sig in cluster])
            }
            for i, cluster in enumerate(clusters)
        ],
        'top_alerts': alerts[:5]  # Top 5 alerts
    }
    
    # Export results
    output_file = analyzer.export_results(results)
    
    # Print summary
    print("\n=== ZVT Neutrino Analysis Summary ===")
    print(f"Total events analyzed: {results['total_events']}")
    print(f"Traditional detections: {sensitivity_results['traditional_detections']}")
    print(f"ZVT-enhanced detections: {sensitivity_results['zvt_detections']}")
    print(f"Sensitivity improvement: {sensitivity_results['improvement_percent']:.1f}%")
    print(f"ZVT clusters found: {results['zvt_clusters']}")
    print(f"Alerts generated: {results['alerts_generated']}")
    print(f"High-priority alerts: {results['high_priority_alerts']}")
    
    if output_file:
        print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
