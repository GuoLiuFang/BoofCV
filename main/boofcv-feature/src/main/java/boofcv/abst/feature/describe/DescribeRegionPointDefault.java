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

package boofcv.abst.feature.describe;

import boofcv.struct.feature.TupleDesc;
import boofcv.struct.image.ImageBase;
import boofcv.struct.image.ImageType;

/**
 * Implements {@link DescribePointGivenRegion} but does nothing. Primarily used for testing.
 *
 * @author Peter Abeles
 */
public class DescribeRegionPointDefault<T extends ImageBase<T>, TD extends TupleDesc<TD>>
		implements DescribePointGivenRegion<T, TD> {

	@Override public void setImage( T image ) {}

	@Override public boolean process( double x, double y, double orientation, double radius, TD description ) {
		return false;
	}

	@Override public boolean isScalable() { return false; }
	@Override public boolean isOriented() { return false; }
	@Override public ImageType<T> getImageType() { return null; }
	@Override public double getCanonicalWidth() { return 0; }
	@Override public TD createDescription() { return null; }
	@Override public Class<TD> getDescriptionType() { return null; }
}
