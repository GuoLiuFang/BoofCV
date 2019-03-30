/*
 * Copyright (c) 2011-2019, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.misc;

import boofcv.generate.AutoTypeImage;
import boofcv.generate.CodeGeneratorBase;
import boofcv.struct.image.ImageType;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import static boofcv.generate.AutoTypeImage.*;


/**
 * Generates functions inside of PixelMath.
 *
 * @author Peter Abeles
 */
public class GeneratePixelMath extends CodeGeneratorBase {

	private AutoTypeImage input;

	ImageType.Family families[] = new ImageType.Family[]{ImageType.Family.GRAY,ImageType.Family.INTERLEAVED};

	@Override
	public void generate() throws FileNotFoundException {
		printPreamble();

		printAbs();
		printNegative();

		List<TwoTemplate> listTwo = new ArrayList<>();
		listTwo.add( new Multiple());
		listTwo.add( new Divide());
		listTwo.add( new Plus());
		listTwo.add( new Minus(true));
		listTwo.add( new Minus(false));

		for( TwoTemplate t : listTwo ) {
			print_img_scalar(t,false);
			print_img_scalar(t,true);
		}

		printAll();
		out.println("}");
	}

	private void printPreamble() {
		out.print("import boofcv.struct.image.*;\n" +
				"\n" +
				"import boofcv.alg.misc.impl.ImplPixelMath_MT;\n" +
				"import boofcv.alg.misc.impl.ImplPixelMath;\n" +
				"import boofcv.concurrency.BoofConcurrency;\n" +
				"import boofcv.alg.InputSanityCheck;\n" +
				"import javax.annotation.Generated;\n" +
				"\n" +
				"/**\n" +
				" * Functions which perform basic arithmetic (e.g. addition, subtraction, multiplication, or " +
				"division) on a pixel by pixel basis.\n" +
				" *\n" +
				" * <p>DO NOT MODIFY: Generated by " + getClass().getName() + ".</p>\n" +
				" *\n" +
				" * @author Peter Abeles\n" +
				" */\n" +
				generatedAnnotation() +
				"public class " + className + " {\n\n");
	}

	public void printAll() {

		AutoTypeImage types[] = AutoTypeImage.getSpecificTypes();

		for( AutoTypeImage t : types ) {
			input = t;

			printBoundImage();
			printDiffAbs();
		}

		AutoTypeImage outputsAdd[] = new AutoTypeImage[]{U16,S16,S32,S32,S32,S64,F32,F64};
		AutoTypeImage outputsSub[] = new AutoTypeImage[]{I16,S16,S32,S32,S32,S64,F32,F64};

		for( int i = 0; i < types.length; i++ ) {
			printAddTwoImages(types[i],outputsAdd[i]);
			printSubtractTwoImages(types[i],outputsSub[i]);

			if( !types[i].isInteger() ) {
				printMultTwoImages(types[i],types[i]);
				printDivTwoImages(types[i],types[i]);
				printLog(types[i],types[i]);
				printLogSign(types[i],types[i]);
				printPow2(types[i], types[i]);
				printSqrt(types[i], types[i]);
			}
		}
	}

	private void print( String funcName , String javadoc , String operation , AutoTypeImage types[] ) {
		for( AutoTypeImage t : types ) {
			input = t;
			for (ImageType.Family family : families) {
				String inputName, columns;
				if (family == ImageType.Family.INTERLEAVED) {
					inputName = input.getInterleavedName();
					columns = "input.width*input.numBands";
				} else {
					inputName = input.getSingleBandName();
					columns = "input.width";
				}

				out.println(javadoc + "\n" +
						"\tpublic static void " + funcName + "( " + inputName + " input , " + inputName + " output ) {\n" +
						"\n" +
						"\t\toutput.reshape(input.width,input.height);\n" +
						"\n" +
						"\t\tint columns = " + columns + ";\n" +
						"\t\tif(BoofConcurrency.USE_CONCURRENT ) {\n" +
						"\t\t\tImplPixelMath_MT."+funcName+"(input.data, input.startIndex, input.stride,\n" +
						"\t\t\t\t\toutput.data, output.startIndex, output.stride,\n" +
						"\t\t\t\t\tinput.height, columns);\n" +
						"\t\t} else {\n" +
						"\t\t\tImplPixelMath."+funcName+"(input.data, input.startIndex, input.stride,\n" +
						"\t\t\t\t\toutput.data, output.startIndex, output.stride,\n" +
						"\t\t\t\t\tinput.height, columns);\n" +
						"\t\t}\n" +
						"\t}\n");
			}
		}
	}

	public void printAbs()
	{
		String javaDoc = "\t/**\n" +
				"\t * Sets each pixel in the output image to be the absolute value of the input image.\n" +
				"\t * Both the input and output image can be the same instance.\n" +
				"\t * \n" +
				"\t * @param input The input image. Not modified.\n" +
				"\t * @param output Where the absolute value image is written to. Modified.\n" +
				"\t */";

		print("abs",javaDoc,"Math.abs(input[indexSrc])",AutoTypeImage.getSigned());
	}

	public void printNegative()
	{
		String javaDoc = "\t/**\n" +
				"\t * Changes the sign of every pixel in the image: output[x,y] = -input[x,y]\n" +
				"\t *\n" +
				"\t * @param input The input image. Not modified.\n" +
				"\t * @param output Where the negated image is written to. Modified.\n" +
				"\t */";

		print("negative",javaDoc,"-input[indexSrc]",AutoTypeImage.getSigned());
	}

	private void print_img_scalar( TwoTemplate template , boolean bounded ) {

		String funcName = template.getName();
		String varName = template.getVariableName();

		for( AutoTypeImage t : template.getTypes() ) {
			input = t;
			String variableType;
			if( template.isScaleOp() )
				variableType = input.isInteger() ? "double" : input.getSumType();
			else
				variableType = input.getSumType();

			String funcArrayName = input.isSigned() ? funcName : funcName+"U";
			funcArrayName += template.isImageFirst() ? "_A" : "_B";

			for (ImageType.Family family : families) {
				String inputName, columns, reshape;
				if (family == ImageType.Family.INTERLEAVED) {
					inputName = input.getInterleavedName();
					columns = "input.width*input.numBands";
					reshape = "output.reshape(input.width,input.height,input.numBands);";
				} else {
					inputName = input.getSingleBandName();
					columns = "input.width";
					reshape = "output.reshape(input.width,input.height);";
				}

				if( bounded ) {
					String sumType = input.getSumType();

					String prototype;
					if (template.isImageFirst()) {
						prototype = "( " + inputName + " input , " + variableType + " " + varName +
								" , " + sumType +" lower , " +sumType+ " upper , " + inputName + " output )";
					} else {
						prototype = "( " + variableType + " " + varName + " , " + inputName + " input , " +
								sumType +" lower , " +sumType+ " upper , " + inputName + " output )";
					}

					out.println(template.getJavaDoc());
					out.println("\tpublic static void " + funcName + prototype + " {\n" +
							"\n" +
							"\t\t" + reshape+ "\n" +
							"\n" +
							"\t\tint columns = " + columns + ";\n" +
							"\t\tif(BoofConcurrency.USE_CONCURRENT ) {\n" +
							"\t\t\tImplPixelMath_MT." + funcArrayName + "(input.data,input.startIndex,input.stride," + varName + ", lower, upper ,\n" +
							"\t\t\t\t\toutput.data,output.startIndex,output.stride,\n" +
							"\t\t\t\t\tinput.height,columns);\n" +
							"\t\t} else {\n" +
							"\t\t\tImplPixelMath." + funcArrayName + "(input.data,input.startIndex,input.stride," + varName + ", lower, upper ,\n" +
							"\t\t\t\t\toutput.data,output.startIndex,output.stride,\n" +
							"\t\t\t\t\tinput.height,columns);\n" +
							"\t\t}\n" +
							"\t}\n");
				} else {
					String prototype;
					if (template.isImageFirst()) {
						prototype = "( " + inputName + " input , " + variableType + " " + varName + " , " + inputName + " output )";
					} else {
						prototype = "( " + variableType + " " + varName + " , " + inputName + " input , " + inputName + " output )";
					}

					out.println(template.getJavaDoc());
					out.println("\tpublic static void " + funcName + prototype + " {\n" +
							"\n" +
							"\t\t" + reshape+ "\n" +
							"\n" +
							"\t\tint columns = " + columns + ";\n" +
							"\t\tif(BoofConcurrency.USE_CONCURRENT ) {\n" +
							"\t\t\tImplPixelMath_MT." + funcArrayName + "(input.data,input.startIndex,input.stride," + varName + " , \n" +
							"\t\t\t\t\toutput.data,output.startIndex,output.stride,\n" +
							"\t\t\t\t\tinput.height,columns);\n" +
							"\t\t} else {\n" +
							"\t\t\tImplPixelMath." + funcArrayName + "(input.data,input.startIndex,input.stride," + varName + " , \n" +
							"\t\t\t\t\toutput.data,output.startIndex,output.stride,\n" +
							"\t\t\t\t\tinput.height,columns);\n" +
							"\t\t}\n" +
							"\t}\n");
				}
			}
		}
	}

	public void printBoundImage() {

		String sumType = input.getSumType();

		out.print("\t/**\n" +
				"\t * Bounds image pixels to be between these two values\n" +
				"\t * \n" +
				"\t * @param img Image\n" +
				"\t * @param min minimum value.\n" +
				"\t * @param max maximum value.\n" +
				"\t */\n" +
				"\tpublic static void boundImage( "+input.getSingleBandName()+" img , "+sumType+" min , "+sumType+" max ) {\n" +
				"\t\tImplPixelMath.boundImage(img,min,max);\n" +
				"\t}\n\n");
	}

	public void printDiffAbs() {
		out.print("\t/**\n" +
				"\t * <p>\n" +
				"\t * Computes the absolute value of the difference between each pixel in the two images.<br>\n" +
				"\t * d(x,y) = |img1(x,y) - img2(x,y)|\n" +
				"\t * </p>\n" +
				"\t * @param imgA Input image. Not modified.\n" +
				"\t * @param imgB Input image. Not modified.\n" +
				"\t * @param output Absolute value of difference image. Can be either input. Modified.\n" +
				"\t */\n" +
				"\tpublic static void diffAbs( "+input.getSingleBandName()+" imgA , "+input.getSingleBandName()+" imgB , "+input.getSingleBandName()+" output ) {\n" +
				"\t\tInputSanityCheck.checkSameShape(imgA,imgB);\n" +
				"\t\toutput.reshape(imgA.width,imgA.height);\n" +
				"\n" +
				"\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\tImplPixelMath_MT.diffAbs(imgA, imgB, output);\n" +
				"\t\t} else {\n" +
				"\t\t\tImplPixelMath.diffAbs(imgA, imgB, output);\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public void printAddTwoImages( AutoTypeImage typeIn , AutoTypeImage typeOut  ) {
		out.print("\t/**\n" +
				"\t * <p>\n" +
				"\t * Performs pixel-wise addition<br>\n" +
				"\t * output(x,y) = imgA(x,y) + imgB(x,y)\n" +
				"\t * </p>\n" +
				"\t * @param imgA Input image. Not modified.\n" +
				"\t * @param imgB Input image. Not modified.\n" +
				"\t * @param output Output image. Can be either input. Modified.\n" +
				"\t */\n" +
				"\tpublic static void add( "+typeIn.getSingleBandName()+" imgA , "+typeIn.getSingleBandName()+" imgB , "+typeOut.getSingleBandName()+" output ) {\n" +
				"\t\tInputSanityCheck.checkSameShape(imgA,imgB);\n" +
				"\t\toutput.reshape(imgA.width,imgA.height);\n" +
				"\n" +
				"\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\tImplPixelMath_MT.add(imgA, imgB, output);\n" +
				"\t\t} else {\n" +
				"\t\t\tImplPixelMath.add(imgA, imgB, output);\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public void printSubtractTwoImages( AutoTypeImage typeIn , AutoTypeImage typeOut ) {
		out.print("\t/**\n" +
				"\t * <p>\n" +
				"\t * Performs pixel-wise subtraction.<br>\n" +
				"\t * output(x,y) = imgA(x,y) - imgB(x,y)\n" +
				"\t * </p>\n" +
				"\t * @param imgA Input image. Not modified.\n" +
				"\t * @param imgB Input image. Not modified.\n" +
				"\t * @param output Output image. Can be either input. Modified.\n" +
				"\t */\n" +
				"\tpublic static void subtract( "+typeIn.getSingleBandName()+" imgA , "+typeIn.getSingleBandName()+" imgB , "
				+typeOut.getSingleBandName()+" output ) {\n" +
				"\t\tInputSanityCheck.checkSameShape(imgA,imgB);\n" +
				"\t\toutput.reshape(imgA.width,imgA.height);\n" +
				"\n" +
				"\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\tImplPixelMath_MT.subtract(imgA, imgB, output);\n" +
				"\t\t} else {\n" +
				"\t\t\tImplPixelMath.subtract(imgA, imgB, output);\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public void printMultTwoImages( AutoTypeImage typeIn , AutoTypeImage typeOut  ) {
		out.print("\t/**\n" +
				"\t * <p>\n" +
				"\t * Performs pixel-wise multiplication<br>\n" +
				"\t * output(x,y) = imgA(x,y) * imgB(x,y)\n" +
				"\t * </p>\n" +
				"\t * @param imgA Input image. Not modified.\n" +
				"\t * @param imgB Input image. Not modified.\n" +
				"\t * @param output Output image. Can be either input. Modified.\n" +
				"\t */\n" +
				"\tpublic static void multiply( "+typeIn.getSingleBandName()+" imgA , "+typeIn.getSingleBandName()+" imgB , "+typeOut.getSingleBandName()+" output ) {\n" +
				"\t\tInputSanityCheck.checkSameShape(imgA,imgB);\n" +
				"\t\toutput.reshape(imgA.width,imgA.height);\n" +
				"\n" +
				"\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\tImplPixelMath_MT.multiply(imgA, imgB, output);\n" +
				"\t\t} else {\n" +
				"\t\t\tImplPixelMath.multiply(imgA, imgB, output);\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public void printDivTwoImages( AutoTypeImage typeIn , AutoTypeImage typeOut  ) {

		out.print("\t/**\n" +
				"\t * <p>\n" +
				"\t * Performs pixel-wise division<br>\n" +
				"\t * output(x,y) = imgA(x,y) / imgB(x,y)\n" +
				"\t * </p>\n" +
				"\t * @param imgA Input image. Not modified.\n" +
				"\t * @param imgB Input image. Not modified.\n" +
				"\t * @param output Output image. Can be either input. Modified.\n" +
				"\t */\n" +
				"\tpublic static void divide( "+typeIn.getSingleBandName()+" imgA , "+typeIn.getSingleBandName()+" imgB , "+typeOut.getSingleBandName()+" output ) {\n" +
				"\t\tInputSanityCheck.checkSameShape(imgA,imgB);\n" +
				"\t\toutput.reshape(imgA.width,imgA.height);\n" +
				"\n" +
				"\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\tImplPixelMath_MT.divide(imgA,imgB,output);\n" +
				"\t\t} else {\n" +
				"\t\t\tImplPixelMath.divide(imgA,imgB,output);\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public void printLog( AutoTypeImage typeIn , AutoTypeImage typeOut ) {
		out.print("\t/**\n" +
				"\t * Sets each pixel in the output image to log( 1 + input(x,y)) of the input image.\n" +
				"\t * Both the input and output image can be the same instance.\n" +
				"\t *\n" +
				"\t * @param input The input image. Not modified.\n" +
				"\t * @param output Where the log image is written to. Modified.\n" +
				"\t */\n" +
				"\tpublic static void log( "+typeIn.getSingleBandName()+" input , "+typeOut.getSingleBandName()+" output ) {\n" +
				"\n" +
				"\t\toutput.reshape(input.width,input.height);\n" +
				"\n" +
				"\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\tImplPixelMath_MT.log(input,output);\n" +
				"\t\t} else {\n" +
				"\t\t\tImplPixelMath.log(input,output);\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public void printLogSign( AutoTypeImage typeIn , AutoTypeImage typeOut ) {
		out.print("\t/**\n" +
				"\t * Sets each pixel in the output image to sgn*log( 1 + sgn*input(x,y)) of the input image.\n" +
				"\t * where sng is the sign of input(x,y). \n" +
				"\t * Both the input and output image can be the same instance.\n" +
				"\t *\n" +
				"\t * @param input The input image. Not modified.\n" +
				"\t * @param output Where the log image is written to. Modified.\n" +
				"\t */\n" +
				"\tpublic static void logSign( "+typeIn.getSingleBandName()+" input , "+typeOut.getSingleBandName()+" output ) {\n" +
				"\n" +
				"\t\toutput.reshape(input.width,input.height);\n" +
				"\n" +
				"\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\tImplPixelMath_MT.logSign(input,output);\n" +
				"\t\t} else {\n" +
				"\t\t\tImplPixelMath.logSign(input,output);\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public void printPow2( AutoTypeImage typeIn , AutoTypeImage typeOut ) {

		out.print("\t/**\n" +
				"\t * Raises each pixel in the input image to the power of two. Both the input and output image can be the \n" +
				"\t * same instance." +
				"\t *\n" +
				"\t * @param input The input image. Not modified.\n" +
				"\t * @param output Where the pow2 image is written to. Can be same as input. Modified.\n" +
				"\t */\n" +
				"\tpublic static void pow2( "+typeIn.getSingleBandName()+" input , "+typeOut.getSingleBandName()+" output ) {\n" +
				"\n" +
				"\t\toutput.reshape(input.width,input.height);\n" +
				"\n" +
				"\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\tImplPixelMath_MT.pow2(input,output);\n" +
				"\t\t} else {\n" +
				"\t\t\tImplPixelMath.pow2(input,output);\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public void printSqrt( AutoTypeImage typeIn , AutoTypeImage typeOut ) {

		out.print("\t/**\n" +
				"\t * Computes the square root of each pixel in the input image. Both the input and output image can be the\n" +
				"\t * same instance.\n" +
				"\t *\n" +
				"\t * @param input The input image. Not modified.\n" +
				"\t * @param output Where the sqrt() image is written to. Can be same as input. Modified.\n" +
				"\t */\n" +
				"\tpublic static void sqrt( "+typeIn.getSingleBandName()+" input , "+typeOut.getSingleBandName()+" output ) {\n" +
				"\n" +
				"\t\toutput.reshape(input.width,input.height);\n" +
				"\n" +
				"\t\tif( BoofConcurrency.USE_CONCURRENT ) {\n" +
				"\t\t\tImplPixelMath_MT.sqrt(input,output);\n" +
				"\t\t} else {\n" +
				"\t\t\tImplPixelMath.sqrt(input,output);\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	class Multiple implements TwoTemplate {

		@Override
		public String getVariableName() { return "value";}

		@Override
		public boolean isScaleOp() { return true; }

		@Override
		public boolean isImageFirst() { return true; }

		@Override
		public AutoTypeImage[] getTypes() { return AutoTypeImage.getSpecificTypes(); }

		@Override
		public String getJavaDoc() {
			return "\t/**\n" +
					"\t * Multiply each element by a scalar value. Both input and output images can\n" +
					"\t * be the same instance.\n" +
					"\t *\n" +
					"\t * @param input The input image. Not modified.\n" +
					"\t * @param value What each element is multiplied by.\n" +
					"\t * @param output The output image. Modified.\n" +
					"\t */";
		}

		@Override
		public String getName() {return "multiply";}

		@Override
		public String getOperation() {
			String round = input.isInteger() ? "Math.round" : "";

			return round+"((input[indexSrc] "+input.getBitWise()+") * value)";
		}
	}

	class Divide implements TwoTemplate {

		@Override public String getVariableName() { return "denominator";}

		@Override public boolean isScaleOp() { return true; }

		@Override public boolean isImageFirst() { return true; }

		@Override public AutoTypeImage[] getTypes() { return AutoTypeImage.getSpecificTypes(); }

		@Override
		public String getJavaDoc() {
			return "\t/**\n" +
					"\t * Divide each element by a scalar value. Both input and output images can be the same instance.\n" +
					"\t *\n" +
					"\t * @param input The input image. Not modified.\n" +
					"\t * @param denominator What each element is divided by.\n" +
					"\t * @param output The output image. Modified.\n" +
					"\t */";
		}

		@Override
		public String getName() {return "divide";}

		@Override
		public String getOperation() {
			String round = input.isInteger() ? "Math.round" : "";

			return round+"((input[indexSrc] "+input.getBitWise()+") / denominator)";
		}
	}

	class Plus implements TwoTemplate {

		@Override public String getVariableName() { return "value";}

		@Override public boolean isScaleOp() { return false; }

		@Override public boolean isImageFirst() { return true; }

		@Override public AutoTypeImage[] getTypes() { return AutoTypeImage.getSpecificTypes(); }

		@Override
		public String getJavaDoc() {
			return "\t/**\n" +
					"\t * Adds a scalar value to each element. Both input and output images can be the same instance.\n" +
					"\t *\n" +
					"\t * @param input The input image. Not modified.\n" +
					"\t * @param value What is added to each element.\n" +
					"\t * @param output The output image. Modified.\n" +
					"\t */";
		}

		@Override
		public String getName() {return "plus";}

		@Override
		public String getOperation() {
			return "((input[indexSrc] "+input.getBitWise()+") + value)";
		}
	}

	class Minus implements TwoTemplate {

		boolean imageFirst;

		public Minus(boolean imageFirst) {
			this.imageFirst = imageFirst;
		}

		@Override public String getVariableName() { return "value";}

		@Override public boolean isScaleOp() { return false; }

		@Override public boolean isImageFirst() { return imageFirst; }

		@Override public AutoTypeImage[] getTypes() { return AutoTypeImage.getSpecificTypes(); }

		@Override
		public String getJavaDoc() {
			if( imageFirst )
				return "\t/**\n" +
						"\t * Subtracts a scalar value from each element. Both input and output images can be the same instance.\n" +
						"\t *\n" +
						"\t * @param input The input image. Not modified.\n" +
						"\t * @param value What is subtracted from each element.\n" +
						"\t * @param output The output image. Modified.\n" +
						"\t */";
			else
				return "\t/**\n" +
						"\t * Subtracts each element's value from a scalar. Both input and output images can be the same instance.\n" +
						"\t *\n" +
						"\t * @param value Scalar value\n" +
						"\t * @param input The input image. Not modified.\n" +
						"\t * @param output The output image. Modified.\n" +
						"\t */";
		}

		@Override
		public String getName() {return "minus";}

		@Override
		public String getOperation() {
			if( imageFirst )
				return "((input[indexSrc] "+input.getBitWise()+") - value)";
			else
				return "(value - (input[indexSrc] "+input.getBitWise()+"))";
		}
	}

	interface Template {
		String getJavaDoc();

		String getName();

		String getOperation();
	}

	interface TwoTemplate extends Template {
		String getVariableName();

		boolean isScaleOp();

		boolean isImageFirst();

		AutoTypeImage[] getTypes();
	}

	public static void main( String[] args ) throws FileNotFoundException {
		GeneratePixelMath gen = new GeneratePixelMath();
		gen.parseArguments(args);
		gen.generate();
	}
}
