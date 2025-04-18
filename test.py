from simulation.security_sim import SecuritySimulation
from simulation.evaluator import SecurityEvaluator
from hardware_controller import HardwareController
from adaptive_mfa import AdaptiveMFA
import logging

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SecurityEvaluation")
    
    # Initialize components
    hardware = HardwareController()
    mfa = AdaptiveMFA(hardware)
    
    # Create simulation
    simulation = SecuritySimulation(hardware, mfa, duration_hours=24)
    
    # Create evaluator
    evaluator = SecurityEvaluator()
    
    # Define algorithms to test
    algorithms = [
        "epsilon_greedy",
        "thompson",
        "ucb",
        "fixed_weights"  # Baseline comparison
    ]
    
    # Run comparison
    logger.info("Starting algorithm comparison")
    results = evaluator.run_comparison(simulation, algorithms)
    
    # Generate plots and save results
    evaluator.plot_results(results)
    logger.info("Evaluation complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    main()