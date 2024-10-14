import data_loader
import model_trainer
import attack_generator
import evaluator
import results_handler

def main():
    # Load and prepare data
    ratings, trainset, testset = data_loader.load_and_prepare_data()

    # Train models
    models = model_trainer.train_models(trainset)

    # Generate attack profiles
    random_attack, average_attack = attack_generator.generate_attacks(ratings)

    # Evaluate models under different scenarios
    results = evaluator.evaluate_all_models(models, testset, random_attack, average_attack)

    # Handle and display results
    results_handler.display_and_save_results(results)

if __name__ == "__main__":
    main()