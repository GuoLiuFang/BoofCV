/*
 * Copyright (c) 2011-2018, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.geo.selfcalib;

import boofcv.alg.geo.GeometricResult;
import boofcv.alg.geo.MultiViewOps;
import boofcv.alg.geo.PerspectiveOps;
import boofcv.struct.calib.CameraPinhole;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Vector3D_F64;
import georegression.struct.se.Se3_F64;
import georegression.struct.se.SpecialEuclideanOps_F64;
import org.ddogleg.struct.FastQueue;
import org.ejml.UtilEjml;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.fixed.CommonOps_DDF3;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.ops.ConvertDMatrixStruct;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Peter Abeles
 */
public class TestSelfCalibrationLinearRotationMulti
	extends CommonAutoCalibrationChecks
{
	CameraPinhole camera = new CameraPinhole(400,410,0,500,520,0,0);
	DMatrixRMaj K = new DMatrixRMaj(3,3);

	@BeforeEach
	public void before() {
		PerspectiveOps.pinholeToMatrix(camera,K);
	}

	@Test
	public void perfect_sk() {
		renderRotationOnly(camera);

		List<Homography2D_F64> viewsI_to_view0 = computeHomographies();

		SelfCalibrationLinearRotationMulti alg = new SelfCalibrationLinearRotationMulti();
		alg.setConstraints(true,false,false,-1);
		assertSame(alg.estimate(viewsI_to_view0), GeometricResult.SUCCESS);

		FastQueue<CameraPinhole> found = alg.getFound();

		for (int i = 0; i < found.size; i++) {
			CameraPinhole f = found.get(i);
			assertEquals(camera.fx,f.fx, UtilEjml.TEST_F64);
			assertEquals(camera.fy,f.fy, UtilEjml.TEST_F64);
			assertEquals(camera.skew,f.skew, UtilEjml.TEST_F64);
			assertEquals(camera.cx,f.cx, UtilEjml.TEST_F64);
			assertEquals(camera.cy,f.cy, UtilEjml.TEST_F64);
		}
	}

	private List<Homography2D_F64> computeHomographies() {
		List<Homography2D_F64> viewsI_to_view0 = new ArrayList<>();

		for (int i = 1; i < listCameraToWorld.size(); i++) {
			Se3_F64 c2w = listCameraToWorld.get(i);
			Vector3D_F64 V = new Vector3D_F64();
//			V.x = rand.nextGaussian();
//			V.y = rand.nextGaussian();
//			V.z = rand.nextGaussian();
//			V.normalize();
			V.z = 1;

			DMatrixRMaj H = MultiViewOps.createHomography(c2w.R,c2w.T,1.1,V,K);
			Homography2D_F64 HH = new Homography2D_F64();
			ConvertDMatrixStruct.convert(H,HH);
			viewsI_to_view0.add(HH);
		}
		return viewsI_to_view0;
	}

	@Test
	public void perfect_pz() {
		fail("Implement");
	}

	@Test
	public void perfect_pz_ka() {
		fail("Implement");
	}

	@Test
	public void perfect_zk_pz_ka() {
		fail("Implement");
	}

	@Test
	public void fillInConstraintMatrix() {
		renderRotationOnly(camera);
		List<Homography2D_F64> viewsI_to_view0 = computeHomographies();
		SelfCalibrationLinearRotationMulti alg = new SelfCalibrationLinearRotationMulti();
		alg.setConstraints(true,false,false,1);
		alg.computeInverseH(viewsI_to_view0);
		alg.fillInConstraintMatrix();
		alg.A.print();

		// Compute 'w' which should be in the null space
		DMatrixRMaj w = new DMatrixRMaj(3,3);
		CommonOps_DDRM.multTransB(K,K,w);
		CommonOps_DDRM.invert(w);
		CommonOps_DDRM.divide(w,w.get(2,2));

		DMatrixRMaj x = new DMatrixRMaj(6,1);
		x.data[0] = w.get(0,0);
		x.data[1] = w.get(0,1);
		x.data[2] = w.get(0,2);
		x.data[3] = w.get(1,1);
		x.data[4] = w.get(1,2);
		x.data[5] = w.get(2,2);

		DMatrixRMaj b = new DMatrixRMaj(6,1);
		CommonOps_DDRM.mult(alg.A,x,b);

		for (int i = 0; i < 6; i++) {
			assertEquals(0,b.get(i,0), UtilEjml.TEST_F64);
		}
	}

	@Test
	public void numberOfConstraints() {
		SelfCalibrationLinearRotationMulti alg = new SelfCalibrationLinearRotationMulti();
		assertEquals(0,alg.numberOfConstraints());

		alg.setConstraints(true,false,false,0);
		assertEquals(1,alg.numberOfConstraints());

		alg.setConstraints(false,true,false,0);
		assertEquals(2,alg.numberOfConstraints());

		alg.setConstraints(false,true,true,1);
		assertEquals(3,alg.numberOfConstraints());

		alg.setConstraints(true,true,true,1);
		assertEquals(4,alg.numberOfConstraints());
	}

	@Test
	public void convertW() {
		double tol = 0.001;

		Homography2D_F64 K = new Homography2D_F64(300,1,320,0,305,330,0,0,1);
		Homography2D_F64 W = new Homography2D_F64();
		CommonOps_DDF3.multTransA(K,K,W);
		CommonOps_DDF3.invert(W,W);
		CameraPinhole found = new CameraPinhole();

		SelfCalibrationLinearRotationMulti alg = new SelfCalibrationLinearRotationMulti();
		alg.setConstraints(false,false,false,-1);
		alg.convertW(W,found);
		assertEquals(K.a11,found.fx, tol);
		assertEquals(K.a12,found.skew, tol);
		assertEquals(K.a13,found.cx, tol);
		assertEquals(K.a22,found.fy, tol);
		assertEquals(K.a23,found.cy, tol);

		// Add zero Skew
		K.a12 = 0;
		CommonOps_DDF3.multTransA(K,K,W);
		CommonOps_DDF3.invert(W,W);
		alg.setConstraints(true,false,false,-1);
		alg.convertW(W,found);
		assertEquals(K.a11,found.fx, tol);
		assertEquals(0,found.skew, tol);
		assertEquals(K.a13,found.cx, tol);
		assertEquals(K.a22,found.fy, tol);
		assertEquals(K.a23,found.cy, tol);

		// Add origin at zero
		K.a13 = K.a23 = 0;
		CommonOps_DDF3.multTransA(K,K,W);
		CommonOps_DDF3.invert(W,W);
		alg.setConstraints(true,true,false,-1);
		alg.convertW(W,found);
		assertEquals(K.a11,found.fx, tol);
		assertEquals(0,found.skew, tol);
		assertEquals(0,found.cx, tol);
		assertEquals(K.a22,found.fy, tol);
		assertEquals(0,found.cy, tol);

		// Add known aspect ratio
		double aspect = K.a11/K.a22;
		CommonOps_DDF3.multTransA(K,K,W);
		CommonOps_DDF3.invert(W,W);
		alg.setConstraints(true,true,true,aspect);
		alg.convertW(W,found);
		assertEquals(K.a11,found.fx, tol);
		assertEquals(0,found.skew, tol);
		assertEquals(0,found.cx, tol);
		assertEquals(K.a22,found.fy, tol);
		assertEquals(0,found.cy, tol);

		// sanity check. give it a bad skew and see what happens
		alg.setConstraints(true,true,true,aspect*aspect);
		alg.convertW(W,found);
		assertTrue(Math.abs(K.a22-found.fy) > tol);
	}

	/**
	 * Test it by computing W0 from W1, then compute W1 again.
	 */
	@Test
	public void extractCalibration() {
		// camera calibration matrix
		Homography2D_F64 K1 = new Homography2D_F64(300,1,320,0,305,330,0,0,1);
		// W = K*K'
		Homography2D_F64 W1 = new Homography2D_F64();
		CommonOps_DDF3.multTransA(K1,K1,W1);
		CommonOps_DDF3.invert(W1,W1);

		Se3_F64 v1_to_v0 = SpecialEuclideanOps_F64.eulerXyz(1,0,0.1,0.1,0.05,-0.09,null);
		DMatrixRMaj H = MultiViewOps.createHomography(v1_to_v0.R,v1_to_v0.T,1,new Vector3D_F64(0,0,1));
		Homography2D_F64 H1 = new Homography2D_F64();
		Homography2D_F64 H1_inv = new Homography2D_F64();
		ConvertDMatrixStruct.convert(H,H1);
		CommonOps_DDF3.invert(H1,H1_inv);

		Homography2D_F64 tmp = new Homography2D_F64();
		Homography2D_F64 W0 = new Homography2D_F64();
		CommonOps_DDF3.multTransA(H1,W1,tmp);
		CommonOps_DDF3.mult(tmp,H1,W0);

		SelfCalibrationLinearRotationMulti alg = new SelfCalibrationLinearRotationMulti();
		alg.setConstraints(false,false,false,-1);
		alg.W0.set(W0);

		CameraPinhole found = new CameraPinhole();
		alg.extractCalibration(H1_inv,found);

		double tol = 0.001;
		assertEquals(K1.a11,found.fx, tol);
		assertEquals(K1.a12,found.skew, tol);
		assertEquals(K1.a13,found.cx, tol);
		assertEquals(K1.a22,found.fy, tol);
		assertEquals(K1.a23,found.cy, tol);
	}
}