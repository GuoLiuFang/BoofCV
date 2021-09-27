/*
 * Copyright (c) 2021, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.filter.convolve.down;

import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.convolve.Kernel1D_S32;
import boofcv.struct.convolve.Kernel2D_F32;
import boofcv.struct.convolve.Kernel2D_S32;
import boofcv.struct.image.*;

import javax.annotation.Generated;

/**
 * Down convolution with kernel renormalization around image borders. Unoptimized naive implementation.
 *
 * <p>DO NOT MODIFY. Automatically generated code created by GenerateConvolveDownNormalizedNaive</p>
 *
 * @author Peter Abeles
 */
@Generated("boofcv.alg.filter.convolve.down.GenerateConvolveDownNormalizedNaive")
public class ConvolveDownNormalizedNaive {

	public static void horizontal( Kernel1D_F32 kernel, GrayF32 input, GrayF32 output, int skip ) {
		output.reshape(input.width/skip, input.height);

		final int radius = kernel.getRadius();

		final int width = input.width - input.width%skip;
		final int height = input.height;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x += skip) {
				float total = 0;
				float div = 0;

				int startX = x - radius;
				int endX = x + radius;

				if (startX < 0) startX = 0;
				if (endX >= input.width) endX = input.width - 1;

				for (int j = startX; j <= endX; j++) {
					float v = kernel.get(j - x + radius);
					total += input.get(j, y)*v;
					div += v;
				}
				output.set(x/skip, y, total/div);
			}
		}
	}

	public static void vertical( Kernel1D_F32 kernel, GrayF32 input, GrayF32 output, int skip ) {
		output.reshape(input.width, input.height/skip);

		final int radius = kernel.getRadius();

		final int width = input.width;
		final int height = input.height - input.height%skip;

		for (int y = 0; y < height; y += skip) {
			for (int x = 0; x < width; x++) {
				float total = 0;
				float div = 0;

				int startY = y - radius;
				int endY = y + radius;

				if (startY < 0) startY = 0;
				if (endY >= input.height) endY = input.height - 1;

				for (int i = startY; i <= endY; i++) {
					float v = kernel.get(i - y + radius);
					total += input.get(x, i)*v;
					div += v;
				}
				output.set(x, y/skip, total/div );
			}
		}
	}

	public static void convolve( Kernel2D_F32 kernel, GrayF32 input, GrayF32 output, int skip ) {
		output.reshape(input.width/skip, input.height/skip);

		final int radius = kernel.getRadius();

		final int width = input.width - input.width%skip;
		final int height = input.height - input.height%skip;

		for (int y = 0; y < height; y += skip) {
			for (int x = 0; x < width; x += skip) {

				int startX = x - radius;
				int endX = x + radius;

				if (startX < 0) startX = 0;
				if (endX >= input.width) endX = input.width - 1;

				int startY = y - radius;
				int endY = y + radius;

				if (startY < 0) startY = 0;
				if (endY >= input.height) endY = input.height - 1;

				float total = 0;
				float div = 0;

				for (int i = startY; i <= endY; i++) {
					for (int j = startX; j <= endX; j++) {
						float v = kernel.get(j - x + radius, i - y + radius);
						total += input.get(j, i)*v;
						div += v;
					}
				}
				output.set(x/skip, y/skip, total/div );
			}
		}
	}

	public static void horizontal( Kernel1D_S32 kernel, GrayU8 input, GrayI8 output, int skip ) {
		output.reshape(input.width/skip, input.height);

		final int radius = kernel.getRadius();

		final int width = input.width - input.width%skip;
		final int height = input.height;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x += skip) {
				int total = 0;
				int div = 0;

				int startX = x - radius;
				int endX = x + radius;

				if (startX < 0) startX = 0;
				if (endX >= input.width) endX = input.width - 1;

				for (int j = startX; j <= endX; j++) {
					int v = kernel.get(j - x + radius);
					total += input.get(j, y)*v;
					div += v;
				}
				output.set(x/skip, y, (total + div/2)/div);
			}
		}
	}

	public static void vertical( Kernel1D_S32 kernel, GrayU8 input, GrayI8 output, int skip ) {
		output.reshape(input.width, input.height/skip);

		final int radius = kernel.getRadius();

		final int width = input.width;
		final int height = input.height - input.height%skip;

		for (int y = 0; y < height; y += skip) {
			for (int x = 0; x < width; x++) {
				int total = 0;
				int div = 0;

				int startY = y - radius;
				int endY = y + radius;

				if (startY < 0) startY = 0;
				if (endY >= input.height) endY = input.height - 1;

				for (int i = startY; i <= endY; i++) {
					int v = kernel.get(i - y + radius);
					total += input.get(x, i)*v;
					div += v;
				}
				output.set(x, y/skip, (total + div/2)/div );
			}
		}
	}

	public static void convolve( Kernel2D_S32 kernel, GrayU8 input, GrayI8 output, int skip ) {
		output.reshape(input.width/skip, input.height/skip);

		final int radius = kernel.getRadius();

		final int width = input.width - input.width%skip;
		final int height = input.height - input.height%skip;

		for (int y = 0; y < height; y += skip) {
			for (int x = 0; x < width; x += skip) {

				int startX = x - radius;
				int endX = x + radius;

				if (startX < 0) startX = 0;
				if (endX >= input.width) endX = input.width - 1;

				int startY = y - radius;
				int endY = y + radius;

				if (startY < 0) startY = 0;
				if (endY >= input.height) endY = input.height - 1;

				int total = 0;
				int div = 0;

				for (int i = startY; i <= endY; i++) {
					for (int j = startX; j <= endX; j++) {
						int v = kernel.get(j - x + radius, i - y + radius);
						total += input.get(j, i)*v;
						div += v;
					}
				}
				output.set(x/skip, y/skip, (total + div/2)/div );
			}
		}
	}

	public static void horizontal( Kernel1D_S32 kernel, GrayS16 input, GrayI16 output, int skip ) {
		output.reshape(input.width/skip, input.height);

		final int radius = kernel.getRadius();

		final int width = input.width - input.width%skip;
		final int height = input.height;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x += skip) {
				int total = 0;
				int div = 0;

				int startX = x - radius;
				int endX = x + radius;

				if (startX < 0) startX = 0;
				if (endX >= input.width) endX = input.width - 1;

				for (int j = startX; j <= endX; j++) {
					int v = kernel.get(j - x + radius);
					total += input.get(j, y)*v;
					div += v;
				}
				output.set(x/skip, y, (total + div/2)/div);
			}
		}
	}

	public static void vertical( Kernel1D_S32 kernel, GrayS16 input, GrayI16 output, int skip ) {
		output.reshape(input.width, input.height/skip);

		final int radius = kernel.getRadius();

		final int width = input.width;
		final int height = input.height - input.height%skip;

		for (int y = 0; y < height; y += skip) {
			for (int x = 0; x < width; x++) {
				int total = 0;
				int div = 0;

				int startY = y - radius;
				int endY = y + radius;

				if (startY < 0) startY = 0;
				if (endY >= input.height) endY = input.height - 1;

				for (int i = startY; i <= endY; i++) {
					int v = kernel.get(i - y + radius);
					total += input.get(x, i)*v;
					div += v;
				}
				output.set(x, y/skip, (total + div/2)/div );
			}
		}
	}

	public static void convolve( Kernel2D_S32 kernel, GrayS16 input, GrayI16 output, int skip ) {
		output.reshape(input.width/skip, input.height/skip);

		final int radius = kernel.getRadius();

		final int width = input.width - input.width%skip;
		final int height = input.height - input.height%skip;

		for (int y = 0; y < height; y += skip) {
			for (int x = 0; x < width; x += skip) {

				int startX = x - radius;
				int endX = x + radius;

				if (startX < 0) startX = 0;
				if (endX >= input.width) endX = input.width - 1;

				int startY = y - radius;
				int endY = y + radius;

				if (startY < 0) startY = 0;
				if (endY >= input.height) endY = input.height - 1;

				int total = 0;
				int div = 0;

				for (int i = startY; i <= endY; i++) {
					for (int j = startX; j <= endX; j++) {
						int v = kernel.get(j - x + radius, i - y + radius);
						total += input.get(j, i)*v;
						div += v;
					}
				}
				output.set(x/skip, y/skip, (total + div/2)/div );
			}
		}
	}

}
