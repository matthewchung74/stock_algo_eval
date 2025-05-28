# uploading local csv files to huggingface
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for HF_TOKEN in environment
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    print("✅ Found HF_TOKEN in environment variables")
    # Login using the token
    try:
        login(token=hf_token)
        print("✅ Successfully logged in to Hugging Face")
    except Exception as e:
        print(f"❌ Error logging in with token: {e}")
        print("Please check your HF_TOKEN in .env file")
        exit(1)
else:
    print("⚠️  HF_TOKEN not found in .env file")
    print("Please add HF_TOKEN=your_token_here to your .env file")
    print("Or run: huggingface-cli login")
    print()

# Dataset configurations
datasets_config = [
    {
        'csv_file': 'NVDA_5min.csv',
        'repo_name': 'matthewchung74/nvda_5_min_bars',
        'description': 'NVIDIA (NVDA) 5-minute stock price bars dataset'
    },
    {
        'csv_file': 'AAPL_5min.csv', 
        'repo_name': 'matthewchung74/aapl_5_min_bars',
        'description': 'Apple (AAPL) 5-minute stock price bars dataset'
    },
    {
        'csv_file': 'ASML_5min.csv',
        'repo_name': 'matthewchung74/asml_5_min_bars', 
        'description': 'ASML 5-minute stock price bars dataset'
    }
]

def upload_dataset(csv_file, repo_name, description):
    """Upload a single CSV file as a Hugging Face dataset"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {csv_file} -> {repo_name}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        return False
    
    try:
        # Load the CSV file
        print(f"📖 Loading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Display basic info about the dataset
        print(f"📊 Dataset Info:")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Show sample data
        print(f"\n📋 Sample data:")
        print(df.head())
        
        # Convert to Hugging Face Dataset
        print(f"\n🔄 Converting to Hugging Face Dataset...")
        dataset = Dataset.from_pandas(df)
        
        # Upload to Hugging Face Hub
        print(f"⬆️  Uploading to {repo_name}...")
        dataset.push_to_hub(
            repo_id=repo_name,
            private=False,  # Set to True if you want private datasets
            commit_message=f"Upload {csv_file} dataset"
        )
        
        print(f"✅ Successfully uploaded {csv_file} to {repo_name}")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{repo_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading {csv_file}: {str(e)}")
        return False

def main():
    """Main function to upload all datasets"""
    
    print("🚀 Starting Hugging Face Dataset Upload Process")
    print("=" * 60)
    
    successful_uploads = 0
    failed_uploads = 0
    
    for config in datasets_config:
        success = upload_dataset(
            csv_file=config['csv_file'],
            repo_name=config['repo_name'], 
            description=config['description']
        )
        
        if success:
            successful_uploads += 1
        else:
            failed_uploads += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📈 UPLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful uploads: {successful_uploads}")
    print(f"❌ Failed uploads: {failed_uploads}")
    print(f"📊 Total datasets: {len(datasets_config)}")
    
    if successful_uploads > 0:
        print(f"\n🎉 Successfully uploaded datasets:")
        for config in datasets_config:
            if os.path.exists(config['csv_file']):
                print(f"   - {config['repo_name']}")
    
    if failed_uploads > 0:
        print(f"\n⚠️  Please check the errors above and try again for failed uploads.")
    
    print(f"\n💡 Tips:")
    print(f"   - Make sure you're logged in: huggingface-cli login")
    print(f"   - Check your internet connection")
    print(f"   - Verify repository names are correct")

if __name__ == "__main__":
    main()
