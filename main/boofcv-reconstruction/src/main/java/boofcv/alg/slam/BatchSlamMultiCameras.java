/*
 * Copyright (c) 2022, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.slam;

import boofcv.alg.structure.LookUpSimilarImages;
import boofcv.alg.structure.PairwiseImageGraph;
import boofcv.alg.structure.SceneWorkingGraph;
import boofcv.misc.BoofMiscOps;
import org.ddogleg.struct.DogArray;
import org.ddogleg.struct.DogArray_B;
import org.ddogleg.struct.FastAccess;
import org.ddogleg.struct.VerbosePrint;
import org.jetbrains.annotations.Nullable;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Batch Simultaneous Location and Mapping (SLAM) system which assumed a known multi camera system is viewing the world.
 * A feature based approach is used were first salient image features are found then matched to each other using
 * descriptors.
 *
 * @author Peter Abeles
 */
public class BatchSlamMultiCameras implements VerbosePrint {

	// Handling known baseline in multi-camera systems
	// - TLDR for now ignore this constraint
	// - TODO How to handle that the baseline between cameras in multi-camera system is assumed to be known?
	// - TODO if the location of one view is known then the location of all views from camera system is known

	// TODO force it to use simplified camera model for now?

	int countConsideredConnections = 3;

	// Checks to see if two views where captured at the same time by the multi-camera system
	CheckSynchronized checkSynchronized;

	GeneratePairwiseGraphFromMultiCameraSystem generatePairwise;

	DogArray<SeedInfo> seeds = new DogArray<>(SeedInfo::new, SeedInfo::reset);
	DogArray_B viewUsed = new DogArray_B();

	DogArray<SceneWorkingGraph> scenes = new DogArray<>(SceneWorkingGraph::new, SceneWorkingGraph::reset);

	private final List<PairwiseImageGraph.View> valid = new ArrayList<>();

	MultiCameraSystem sensors;
	ViewToCamera viewToCamera;

	// If a scene as a known scale, which means the seed has a pair of views with a known extrinsic relationship
	boolean activeSceneKnownScale;

	PrintStream verbose;

	public void process( MultiCameraSystem sensors, LookUpSimilarImages similarImages ) {
		// Learn how much geometric information is available between views
		generatePairwise.process(sensors, similarImages);

		// Decide which views are preferred as seeds
		scoreViewsAsSeeds();

		// All views can be used as a seed or added to a scene
		viewUsed.resetResize(seeds.size, true);

		// Select seeds and perform reconstructions
		PairwiseImageGraph pairwise = generatePairwise.getPairwise();
		while (true) {
			SeedInfo seed = selectSeedForScene(pairwise);
			if (seed == null)
				return;

			if (!initializeNewScene(pairwise, seed)) {
				// TODO abort if it fails X times in a row
				continue;
			}

			SceneWorkingGraph scene = scenes.getTail();

			while (scene.open.size > 0) {
				// Select the view with the most informative connections to known views in the scene
				PairwiseImageGraph.View target = selectViewToExpandInto(scene);

				// Couldn't find a valid view to add
				if (target == null)
					break;

				// Try adding this view to the scene
				if (!expandIntoView(scene, target)) {
					continue;
				}

				// TODO sometimes refine the entire scene

			}

			// TODO refine the entire scene
		}
	}

	/**
	 * Score each view as a potential seed.
	 */
	void scoreViewsAsSeeds() {
		FastAccess<PairwiseImageGraph.View> views = generatePairwise.getPairwise().nodes;
		seeds.resetResize(views.size);
		for (int viewIdx = 0; viewIdx < views.size; viewIdx++) {
			PairwiseImageGraph.View v = views.get(viewIdx);
			SeedInfo s = seeds.get(viewIdx);
			s.viewIndex = viewIdx;
			scoreViewAsSeed(v, s);
		}
	}

	/**
	 * Scores a view as a seed based on the scores of the best N connected motion. Motions are selected which are
	 * discintive from other already selected motions
	 */
	void scoreViewAsSeed( PairwiseImageGraph.View target, SeedInfo info ) {
		// Select the motion with the best score
		double bestScore = 0;
		int bestIndex = -1;
		for (int i = 0; i < target.connections.size; i++) {
			PairwiseImageGraph.Motion m = target.connections.get(i);

			if (!m.is3D || m.score3D <= bestScore)
				continue;

			bestScore = m.score3D;
			bestIndex = i;
		}
		// Nothing was selected, this is a horrible seed
		if (bestIndex == -1)
			return;

		addNeighbor(info, target, target.connections.get(bestIndex), bestScore);

		// The remaining neighbors are selected by finding the neighbor which the best score which is the minimum
		// of the score to the target and any of the already connected. This avoids adds two very similar view
		// that happen to have a high score to the target
		for (int considerIdx = 1; considerIdx < countConsideredConnections; considerIdx++) {
			bestScore = 0;
			bestIndex = -1;
			for (int connIdx = 0; connIdx < target.connections.size; connIdx++) {
				PairwiseImageGraph.Motion m = target.connections.get(connIdx);
				PairwiseImageGraph.View mview = m.other(target);

				// Only consider motions which have not been selected and could have a better score
				if (!m.is3D || m.score3D <= bestScore || info.neighbors.contains(m))
					continue;

				double score = m.score3D;
				for (int nghIdx = 0; nghIdx < info.neighbors.size(); nghIdx++) {
					PairwiseImageGraph.View nview = info.neighbors.get(nghIdx).other(target);
					@Nullable PairwiseImageGraph.Motion m2n = mview.findMotion(nview);
					if (m2n == null)
						continue;
					if (m2n.is3D && m2n.score3D < score)
						score = m2n.score3D;
				}

				if (score <= bestScore)
					continue;

				bestScore = score;
				bestIndex = connIdx;
			}

			if (bestIndex == -1)
				return;

			addNeighbor(info, target, target.connections.get(bestIndex), bestScore);
		}
	}

	public @Nullable SeedInfo selectSeedForScene( PairwiseImageGraph pairwise ) {
		for (int seedIdx = 0; seedIdx < seeds.size; seedIdx++) {
			// See if this view can be used
			SeedInfo candidate = seeds.get(seedIdx);
			if (viewUsed.get(candidate.viewIndex))
				continue;

			// Remove views which can't be seeds
			for (int removeIdx = seedIdx; removeIdx >= 0; removeIdx--) {
				seeds.removeSwap(removeIdx);
			}

			return candidate;
		}
		return null;
	}

	public boolean initializeNewScene( PairwiseImageGraph pairwise, SeedInfo seed ) {
		SceneWorkingGraph scene = scenes.grow();

		// Create a camera for every camera in the multi camera system
		for (int i = 0; i < sensors.cameras.size(); i++) {
			scene.addCamera(i);
		}

		activeSceneKnownScale = !seed.knownScale;

		PairwiseImageGraph.View pseed = pairwise.nodes.get(seed.viewIndex);

		if (activeSceneKnownScale) {
			SceneWorkingGraph.Camera camSeed = lookupCamera(scene, pseed);
			SceneWorkingGraph.View wseed = scene.addView(pseed, camSeed);
			SceneWorkingGraph.InlierInfo inliers = wseed.inliers.grow();

			for (int i = 0; i < seed.neighbors.size(); i++) {
				PairwiseImageGraph.Motion m = seed.neighbors.get(i);
				PairwiseImageGraph.View dst = m.other(pseed);
				if (!isExtrinsicsKnown(pseed, dst)) {
					continue;
				}

				// Don't add a view which has already been used again
				if (viewUsed.get(dst.index))
					continue;

				SceneWorkingGraph.Camera camDst = lookupCamera(scene, dst);
				SceneWorkingGraph.View wdst = scene.addView(dst, camDst);

				inliers.views.add(pseed);
				inliers.views.add(dst);
				// TODO add list of observations from each view

				// Mark these two views as being used
				viewUsed.set(pseed.index, true);
				viewUsed.set(dst.index, true);
				break;
			}

			// See if it failed to find a valid view. Probably a bug
			if (scene.listViews.size() == 1)
				return false;

			// Add all neighbors which are connected to the seed to the open list
			for (int i = 0; i < seed.neighbors.size(); i++) {
				PairwiseImageGraph.Motion m = seed.neighbors.get(i);
				PairwiseImageGraph.View dst = m.other(pseed);

				if (viewUsed.get(dst.index))
					continue;

				scene.open.add(dst);
			}
			return true;
		} else {
			throw new RuntimeException("Handle situation where there is no known scale");
		}
	}

	protected @Nullable PairwiseImageGraph.View selectViewToExpandInto( SceneWorkingGraph scene ) {
		int bestIdx = -1;
		double bestScore = 0.0;
		int bestValidCount = 0;

		for (int openIdx = 0; openIdx < scene.open.size; openIdx++) {
			final PairwiseImageGraph.View pview = scene.open.get(openIdx);

			// See which views in the scene pview can connect to
			valid.clear();
			for (int connIdx = 0; connIdx < pview.connections.size; connIdx++) {
				PairwiseImageGraph.Motion m = pview.connections.get(connIdx);
				PairwiseImageGraph.View dst = m.other(pview);
				if (!m.is3D || !scene.isKnown(dst))
					continue;
				valid.add(dst);
			}

			double bestLocalScore = 0.0;
			for (int idx0 = 0; idx0 < valid.size(); idx0++) {
				PairwiseImageGraph.View viewB = valid.get(idx0);
				PairwiseImageGraph.Motion m0 = Objects.requireNonNull(pview.findMotion(viewB));

				for (int idx1 = idx0 + 1; idx1 < valid.size(); idx1++) {
					PairwiseImageGraph.View viewC = valid.get(idx1);
					PairwiseImageGraph.Motion m2 = viewB.findMotion(viewC);

					if (m2 == null || !m2.is3D)
						continue;

					PairwiseImageGraph.Motion m1 = Objects.requireNonNull(pview.findMotion(viewC));

					double s = BoofMiscOps.min(m0.score3D, m1.score3D, m2.score3D);

					bestLocalScore = Math.max(s, bestLocalScore);
				}
			}

			// strongly prefer 3 or more. Technically the above test won't check for this but in the future it will
			// so this test serves as a reminder
			if (Math.min(3, valid.size()) >= bestValidCount && bestLocalScore > bestScore) {
				bestValidCount = Math.min(3, valid.size());
				bestScore = bestLocalScore;
				bestIdx = openIdx;
			}
		}

		if (bestIdx < 0)
			return null;

		return scene.open.removeSwap(bestIdx);
	}

	/**
	 *
	 * @param scene The known scene
	 * @param ptarget The view which is to be added to the scene
	 * @return true if successful and the scene was modified. If false the scene was not modified
	 */
	protected boolean expandIntoView( SceneWorkingGraph scene, PairwiseImageGraph.View ptarget ) {
		// TODO find all views in the scene which are connected to the target

		// TODO see if the extrinsics are known, if so use those

		// TODO if extrinsics are not known, estimate them


		return true;
	}

	private SceneWorkingGraph.Camera lookupCamera( SceneWorkingGraph scene, PairwiseImageGraph.View pview ) {
		return scene.cameras.get(sensors.lookupCamera(viewToCamera.lookup(pview.id)).index);
	}

	public void addNeighbor( SeedInfo info, PairwiseImageGraph.View target, PairwiseImageGraph.Motion m, double score ) {
		info.knownScale |= isExtrinsicsKnown(target, m.other(target));
		info.score += score;
		info.neighbors.add(m);
	}

	/** Returns true if the two views have a known baseline / extrinsics between them */
	public boolean isExtrinsicsKnown( PairwiseImageGraph.View va, PairwiseImageGraph.View vb ) {
		return checkSynchronized.isSynchronized(va.id, vb.id);
	}

	@Override public void setVerbose( @Nullable PrintStream out, @Nullable Set<String> options ) {
		this.verbose = BoofMiscOps.addPrefix(this, out);
	}

	public static class SeedInfo implements Comparable<SeedInfo> {
		int viewIndex;
		double score;
		// This seed will have a known scale if one of the motions associated with it comes from a known stereo pair
		boolean knownScale;
		// List of motions which connect to a neighboring view used to compute the seed's score
		List<PairwiseImageGraph.Motion> neighbors = new ArrayList<>();

		public void reset() {
			viewIndex = -1;
			score = 0;
			knownScale = false;
			neighbors.clear();
		}

		/**
		 * Prefer a seed with a known scale and higher score
		 */
		@Override public int compareTo( BatchSlamMultiCameras.SeedInfo o ) {
			if (knownScale == o.knownScale) {
				return Double.compare(o.score, score);
			} else if (knownScale) {
				return 1;
			} else {
				return 0;
			}
		}
	}
}
