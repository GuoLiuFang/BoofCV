/*
 * Copyright (c) 2011-2012, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.feature.disparity.impl;

import boofcv.alg.InputSanityCheck;
import boofcv.alg.feature.disparity.DisparityScoreSadRect;
import boofcv.alg.feature.disparity.DisparitySelect;
import boofcv.struct.image.ImageFloat32;
import boofcv.struct.image.ImageSingleBand;

/**
 * <p>
 * Implementation of {@link boofcv.alg.feature.disparity.DisparityScoreSadRect} which processes {@limk ImageFloat32} as
 * input images.
 * </p>
 * <p>
 * DO NOT MODIFY. Generated by {@link GenerateDisparityScoreSadRect}.
 * </p>
 * 
 * @author Peter Abeles
 */
public class ImplDisparityScoreSadRect_F32<Disparity extends ImageSingleBand>
	extends DisparityScoreSadRect<ImageFloat32,Disparity>
{

	// Computes disparity from scores
	DisparitySelect<float[],Disparity> computeDisparity;

	// stores the local scores for the width of the region
	float elementScore[];
	// scores along horizontal axis for current block
	// To allow right to left validation all disparity scores are stored for the entire row
	// size = num columns * maxDisparity
	// disparity for column i is stored in elements i*maxDisparity to (i+1)*maxDisparity
	float horizontalScore[][];
	// summed scores along vertical axis
	// This is simply the sum of like elements in horizontal score
	float verticalScore[];

	/**
	 * Configures disparity calculation.
	 *
	 * @param maxDisparity Maximum disparity that it will calculate. Must be > 0
	 * @param regionRadiusX Radius of the rectangular region along x-axis.
	 * @param regionRadiusY Radius of the rectangular region along y-axis.
	 * @param computeDisparity Algorithm which computes the disparity from the score.
	 */
	public ImplDisparityScoreSadRect_F32(int maxDisparity,
										int regionRadiusX, int regionRadiusY,
										DisparitySelect<float[],Disparity> computeDisparity) {
		super(maxDisparity,regionRadiusX,regionRadiusY);

		this.computeDisparity = computeDisparity;
	}

	@Override
	public void process( ImageFloat32 left , ImageFloat32 right , Disparity disparity ) {
		// initialize data structures
		InputSanityCheck.checkSameShape(left,right,disparity);

		lengthHorizontal = left.width*maxDisparity;
		if( horizontalScore == null || verticalScore.length < lengthHorizontal ) {
			horizontalScore = new float[regionHeight][lengthHorizontal];
			verticalScore = new float[lengthHorizontal];
			elementScore = new float[ left.width ];
		}

		computeDisparity.configure(disparity,maxDisparity,radiusX);

		// initialize computation
		computeFirstRow(left, right);
		// efficiently compute rest of the rows using previous results to avoid repeat computations
		computeRemainingRows(left, right);
	}

	/**
	 * Initializes disparity calculation by finding the scores for the initial block of horizontal
	 * rows.
	 */
	private void computeFirstRow(ImageFloat32 left, ImageFloat32 right ) {
		// compute horizontal scores for first row block
		for( int row = 0; row < regionHeight; row++ ) {

			float scores[] = horizontalScore[row];

			computeScoreRow(left, right, row, scores);
		}

		// compute score for the top possible row
		for( int i = 0; i < lengthHorizontal; i++ ) {
			int sum = 0;
			for( int row = 0; row < regionHeight; row++ ) {
				sum += horizontalScore[row][i];
			}
			verticalScore[i] = sum;
		}

		// compute disparity
		computeDisparity.process(radiusY, verticalScore);
	}

	/**
	 * Using previously computed results it efficiently finds the disparity in the remaining rows.
	 * When a new block is processes the last row/column is subtracted and the new row/column is
	 * added.
	 */
	private void computeRemainingRows( ImageFloat32 left, ImageFloat32 right )
	{
		for( int row = regionHeight; row < left.height; row++ ) {
			int oldRow = row%regionHeight;

			// subtract first row from vertical score
			float scores[] = horizontalScore[oldRow];
			for( int i = 0; i < lengthHorizontal; i++ ) {
				verticalScore[i] -= scores[i];
			}

			computeScoreRow(left, right, row, scores);

			// add the new score
			for( int i = 0; i < lengthHorizontal; i++ ) {
				verticalScore[i] += scores[i];
			}

			// compute disparity
			computeDisparity.process(row - regionHeight + 1 + radiusY, verticalScore);
		}
	}

	/**
	 * Computes disparity score for an entire row.
	 *
	 * For a given disparity, the score for each region on the left share many components in common.
	 * Because of this the scores are computed with disparity being the outer most loop
	 *
	 * @param left left image
	 * @param right Right image
	 * @param row Image row being examined
	 * @param scores Storage for disparity scores.
	 */
	protected void computeScoreRow(ImageFloat32 left, ImageFloat32 right, int row, float[] scores) {

		// disparity as the outer loop to maximize common elements in inner loops, reducing redundant calculations
		for( int d = 0; d < maxDisparity; d++ ) {
			final int elementMax = left.width-d;
			final int scoreMax = elementMax-regionWidth;
			int indexScore = left.width*d+d;

			int indexLeft = left.startIndex + left.stride*row + d;
			int indexRight =  right.startIndex + right.stride*row;

			// Fill elementScore with all the scores for this row at disparity d
			computeScoreRow(left, right, elementMax, indexLeft, indexRight);

			// score at the first column
			float score = 0;
			for( int i = 0; i < regionWidth; i++ )
				score += elementScore[i];

			scores[indexScore++] = score;

			// scores for the remaining columns
			for( int col = 0; col < scoreMax; col++ , indexScore++ ) {
				scores[indexScore] = score += elementScore[col+regionWidth] - elementScore[col];
			}
		}
	}

	/**
	 * compute the score for each element all at once to encourage the JVM to optimize and
	 * encourage the JVM to optimize this section of code.
	 *
	 * Was original inline, but was actually slightly slower by about 3% consistently,  It
	 * is in its own function so that it can be overridden and have different cost functions
	 * inserted easily.
	 */
	protected void computeScoreRow(ImageFloat32 left, ImageFloat32 right,
								   int elementMax, int indexLeft, int indexRight)
	{
		for( int rCol = 0; rCol < elementMax; rCol++ ) {
			float diff = (left.data[ indexLeft++ ]) - (right.data[ indexRight++ ]);

			elementScore[rCol] = Math.abs(diff);
		}
	}

	@Override
	public Class<ImageFloat32> getInputType() {
		return ImageFloat32.class;
	}

	@Override
	public Class<Disparity> getDisparityType() {
		return computeDisparity.getDisparityType();
	}

}
