
def collect_text_lengths_by_model():
    """Collect whole text lengths (character and word count) for all models."""
    model_text_lengths = {
        'word_counts': defaultdict(list),
        'char_counts': defaultdict(list)
    }
    
    # Process Gemini models
    gemini_data = load_jsonl_data('data/text/code_gen_data/gemini_specific.jsonl')
    for item in gemini_data:
        if 'content' in item and 'label_detailed' in item and item['label_detailed'] in GEMINI_MODEL:
            model = item['label_detailed']
            text = item['content']
            
            # Count words in entire text
            words = re.findall(r'\S+', text)
            model_text_lengths['word_counts'][model].append(len(words))
            
            # Count characters in entire text
            model_text_lengths['char_counts'][model].append(len(text))
    
    # Process other models
    other_data = load_jsonl_data('data/text/labels/AI/thesis_AI_en.jsonl')
    for item in other_data:
        if 'content' in item and 'label_detailed' in item and item['label_detailed'] in OTHER_MODELS:
            model = item['label_detailed']
            text = item['content']
            
            # Count words in entire text
            words = re.findall(r'\S+', text)
            model_text_lengths['word_counts'][model].append(len(words))
            
            # Count characters in entire text
            model_text_lengths['char_counts'][model].append(len(text))
    
    return model_text_lengths

def visualize_text_lengths(model_text_lengths):
    """Create visualizations for text length metrics."""
    # Create separate figures for word counts and character counts
    metrics = ['word_counts', 'char_counts']
    titles = ['Text Length Distribution (Words)', 'Text Length Distribution (Characters)']
    filenames = ['text_length_words.png', 'text_length_chars.png']
    
    for metric, title, filename in zip(metrics, titles, filenames):
        plt.figure(figsize=(16, 9))
        
        # Get the data for this metric
        data = model_text_lengths[metric]
        
        # Find max value for bin sizing
        max_value = max(max(lengths) for lengths in data.values() if lengths)
        
        # Create appropriate bins based on the data range
        if metric == 'word_counts':
            bin_size = 50
            if max_value > 1000:
                bins = list(range(0, 500, bin_size)) + list(range(500, 1000, 100)) + list(range(1000, max_value + 200, 200))
            else:
                bins = range(0, max_value + bin_size, bin_size)
        else:  # character counts
            bin_size = 250
            if max_value > 5000:
                bins = list(range(0, 2500, bin_size)) + list(range(2500, 5000, 500)) + list(range(5000, max_value + 1000, 1000))
            else:
                bins = range(0, max_value + bin_size, bin_size)
        
        # Plot histogram for each model
        for model, color in ALL.items():
            if model in data and data[model]:
                plt.hist(
                    data[model],
                    bins=bins,
                    alpha=0.7,
                    color=color,
                    label=f"{model} (avg: {np.mean(data[model]):.1f})",
                    edgecolor='black',
                    linewidth=0.5,
                    rwidth=0.85  # Make bars wider
                )
        
        # Add chart labels and formatting
        unit = "Words" if metric == 'word_counts' else "Characters" 
        plt.xlabel(f'Text Length ({unit})', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats_text = "Model Statistics:\n" + "-" * 40 + "\n"
        for model in ALL:
            if model in data and data[model]:
                avg = np.mean(data[model])
                median = np.median(data[model])
                max_len = max(data[model])
                min_len = min(data[model])
                stats_text += f"{model}:\n  Avg={avg:.1f}\n  Med={median:.1f}\n  Range={min_len}-{max_len}\n\n"
        
        # Place the stats box
        plt.figtext(0.75, 0.25, stats_text, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        
    return plt

def print_text_length_statistics(model_text_lengths):
    """Print statistics about the text length data."""
    print("\nText Length Statistics by Model:")
    print("=" * 100)
    print(f"{'Model':<25} {'Count':<8} {'Min Words':<10} {'Max Words':<10} {'Avg Words':<10} {'Min Chars':<10} {'Max Chars':<10} {'Avg Chars':<10}")
    print("-" * 100)
    
    for model in sorted(ALL.keys()):
        word_counts = model_text_lengths['word_counts'].get(model, [])
        char_counts = model_text_lengths['char_counts'].get(model, [])
        
        if not word_counts or not char_counts:
            continue
            
        print(f"{model:<25} {len(word_counts):<8} " 
              f"{min(word_counts):<10} {max(word_counts):<10} {np.mean(word_counts):.1f}{'':>5} "
              f"{min(char_counts):<10} {max(char_counts):<10} {np.mean(char_counts):.1f}")

def main():
    print("Collecting sentence lengths by model...")
    model_sentences = collect_sentence_lengths_by_model()
    
    print("Collecting whole text lengths by model...")
    model_text_lengths = collect_text_lengths_by_model()
    
    # Print statistics on data collection
    for model, sentences in model_sentences.items():
        print(f"Collected {len(sentences)} sentences for {model}")
    
    if not any(model_sentences.values()):
        print("No data found. Check file paths and model names.")
        return
    
    print("\nCreating sentence length histogram...")
    plt1 = create_combined_histogram(model_sentences)
    
    print("\nCreating text length visualizations...")
    plt2 = visualize_text_lengths(model_text_lengths)
    
    print("\nSentence Length Statistics by Model:")
    # ... [rest of the existing statistics code]
    
    print_range_statistics(model_sentences)
    
    print_text_length_statistics(model_text_lengths)
    
    print("\nAnalysis complete. Visualizations saved as PNG files.")
    plt1.show()
    
if __name__ == "__main__":
    main()