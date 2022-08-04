#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Getting artifact path via wandb")
    input_name = args.input_artifact
    artifact_local_path = wandb.use_artifact(input_name).file()
    
    logger.info("Reading dataframe")
    artifact_frame = pd.read_csv(artifact_local_path)

    logger.info("Removing Outliers in the price column")
    # Drop outliers 
    min_price_ = args.min_price
    max_price_ = args.max_price
    idx = artifact_frame['price'].between(min_price_, max_price_)
    artifact_frame = artifact_frame[idx].copy()

    # Convert last_review to datetime
    logger.info("Convert last review column to datetime format")
    artifact_frame['last_review'] = pd.to_datetime(artifact_frame['last_review'])

    logger.info("Saving the pre-processing data")
    outfile = args.output_artifact
    artifact_frame.to_csv(outfile, index = False)

    logger.info("Generating the output artifact")
    artifact = wandb.Artifact(
        name = args.output_artifact,
        type = args.output_type,
        description = args.output_description
    )
    logger.info("Uploading & Logging the artifact into the Wandb")

    artifact.add_file(outfile)
    run.log_artifact(artifact)

    logger.info("Finishing the pre-processing")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning",
                                    fromfile_prefix_chars="@")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price value",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price value",
        required=True
    )


    args = parser.parse_args()

    go(args)
