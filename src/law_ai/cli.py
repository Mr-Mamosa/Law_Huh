import argparse
from .core.engine import QueryEngine
import json

def main():
    parser = argparse.ArgumentParser(description='Asks a question to the Legal AI Query Engine.')
    parser.add_argument('query', type=str, help='The question to ask the AI.')
    parser.add_argument('--analyze-precedent', action='store_true', help='Analyze precedent chain.')
    parser.add_argument('--verify-claims', action='store_true', help='Verify claims.')
    args = parser.parse_args()

    print(f'Querying AI with: "{args.query}"')

    try:
        engine = QueryEngine()
        response = engine.ask_api(
            args.query,
            analyze_precedent=args.analyze_precedent,
            verify_claims=args.verify_claims
        )

        print("\n--- AI Response ---")
        print(f"\nAnswer:\n{response.get('answer')}\n")
        print(f"Risk Score: {response.get('risk_score')}")
        print(f"Verified Claims: {response.get('verified_claims')}/{response.get('total_claims')}")
        print(f"Tags: {', '.join(response.get('tags', []))}")
        print(f"Execution Time: {response.get('execution_time')} seconds")
        if response.get('precedent_data'):
            print("\nPrecedent Data:")
            print(json.dumps(response['precedent_data'], indent=2))

    except Exception as e:
        print(f"An error occurred while querying the AI: {e}")

if __name__ == "__main__":
    main()