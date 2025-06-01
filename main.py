from multiprocessing import Pool, cpu_count
from itertools import combinations
import math
import time

# GCD Calculator
def calculate_gcd_filtered(pair):
    a, b = pair
    gcd = math.gcd(a, b)
    if gcd > 1:
        return (a, b, gcd)
    return None

# Read numbers from .txt file
def read_num_file(file):
    with open(file, 'r') as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]

# Writes per file's chunk
def write_file(buffer, file):
    with open(file, 'a') as f:
        for a, b, mcd in buffer:
            f.write(f"{a},{b},{mcd}\n")

if __name__ == '__main__':
    start_time = time.time()

    nums = read_num_file("integers.txt")
    total_pairs = len(nums) * (len(nums) - 1) // 2
    print(f"Processing approximately {total_pairs:,} pairs...")

    generator = combinations(nums, 2)
    output_file = "pairs_gcd_greater_1.txt"

    # Erase previous file if exists
    open(output_file, 'w').close()

    buffer = []
    buffer_size = 10000
    procesed_pairs = 0
    saved_pairs = 0

    with Pool(cpu_count()) as pool:
        for result in pool.imap(calculate_gcd_filtered, generator, chunksize=5000):
            procesed_pairs += 1
            if result:
                buffer.append(result)
                saved_pairs += 1

            # Writes in disk per chunk
            if len(buffer) >= buffer_size:
                write_file(buffer, output_file)
                buffer.clear()

            # Show process stats every 10 million pairs
            if procesed_pairs % 10_000_000 == 0:
                elapsed = time.time() - start_time
                print(f"{procesed_pairs:,} Processed pairs — {saved_pairs:,} saved — {elapsed/60:.2f} min")

    # Write last chunk
    if buffer:
        write_file(buffer, output_file)

    total_time = time.time() - start_time
    print(f"\nFinished in {total_time/60:.2f} minutes")
    print(f"Total pairs processed: {procesed_pairs:,}")
    print(f"Pairs with GCD > 1: {saved_pairs:,}")
