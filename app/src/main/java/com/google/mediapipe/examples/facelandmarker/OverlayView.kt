package com.google.mediapipe.examples.facelandmarker

/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.Drawable
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    // Enum to track which filter is active
    enum class FilterType {
        NONE,
        JOKER_MASK,
        GLASSES
    }

    private var results: FaceLandmarkerResult? = null
    private var linePaint = Paint()
    private var pointPaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    private var jokerMaskBitmap: Bitmap? = null
    private var glassesBitmap: Bitmap? = null
    
    // Control whether to show landmarks and connectors
    private var showLandmarks: Boolean = true
    
    // Current active filter
    private var currentFilter: FilterType = FilterType.JOKER_MASK

    init {
        initPaints()
        // Load the filter bitmaps from resources
        context?.let {
            try {
                jokerMaskBitmap = BitmapFactory.decodeResource(it.resources, R.drawable.joker_mask)
                
                // Properly load vector drawable as bitmap
                glassesBitmap = drawableToBitmap(ContextCompat.getDrawable(it, R.drawable.glasses))
                
                Log.d(TAG, "Joker mask loaded: ${jokerMaskBitmap != null}")
                Log.d(TAG, "Glasses loaded: ${glassesBitmap != null}")
            } catch (e: Exception) {
                Log.e(TAG, "Error loading filter images: ${e.message}")
            }
        }
    }
    
    // Helper function to convert drawable to bitmap
    private fun drawableToBitmap(drawable: Drawable?): Bitmap? {
        if (drawable == null) {
            Log.e(TAG, "Drawable is null")
            return null
        }
        
        if (drawable is BitmapDrawable) {
            return drawable.bitmap
        }
        
        // Create bitmap with transparent pixels
        val bitmap = Bitmap.createBitmap(
            drawable.intrinsicWidth,
            drawable.intrinsicHeight,
            Bitmap.Config.ARGB_8888
        )
        
        val canvas = Canvas(bitmap)
        drawable.setBounds(0, 0, canvas.width, canvas.height)
        drawable.draw(canvas)
        
        return bitmap
    }

    fun clear() {
        results = null
        linePaint.reset()
        pointPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }
    
    // Method to control landmark visibility
    fun setShowLandmarks(show: Boolean) {
        showLandmarks = show
        invalidate()
    }
    
    // Method to switch between filters
    fun setFilterType(filterType: FilterType) {
        currentFilter = filterType
        Log.d(TAG, "Filter set to: $filterType")
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // Clear previous drawings if results exist but have no face landmarks
        if (results?.faceLandmarks().isNullOrEmpty()) {
            clear()
            return
        }

        results?.let { faceLandmarkerResult ->

            // Calculate scaled image dimensions
            val scaledImageWidth = imageWidth * scaleFactor
            val scaledImageHeight = imageHeight * scaleFactor

            // Calculate offsets to center the image on the canvas
            val offsetX = (width - scaledImageWidth) / 2f
            val offsetY = (height - scaledImageHeight) / 2f

            // Iterate through each detected face
            faceLandmarkerResult.faceLandmarks().forEach { faceLandmarks ->
                // Draw landmarks and connectors only if showLandmarks is true
                if (showLandmarks) {
                    // Draw all landmarks for the current face
                    drawFaceLandmarks(canvas, faceLandmarks, offsetX, offsetY)

                    // Draw all connectors for the current face
                    drawFaceConnectors(canvas, faceLandmarks, offsetX, offsetY)
                }
                
                // Draw the selected filter
                when (currentFilter) {
                    FilterType.JOKER_MASK -> drawJokerMask(canvas, faceLandmarks, offsetX, offsetY)
                    FilterType.GLASSES -> drawGlasses(canvas, faceLandmarks, offsetX, offsetY)
                    FilterType.NONE -> {} // No filter
                }
            }
        }
    }

    /**
     * Draws all landmarks for a single face on the canvas.
     */
    private fun drawFaceLandmarks(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        faceLandmarks.forEach { landmark ->
            val x = landmark.x() * imageWidth * scaleFactor + offsetX
            val y = landmark.y() * imageHeight * scaleFactor + offsetY
            canvas.drawPoint(x, y, pointPaint)
        }
    }

    /**
     * Draws all the connectors between landmarks for a single face on the canvas.
     */
    private fun drawFaceConnectors(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        FaceLandmarker.FACE_LANDMARKS_CONNECTORS.filterNotNull().forEach { connector ->
            val startLandmark = faceLandmarks.getOrNull(connector.start())
            val endLandmark = faceLandmarks.getOrNull(connector.end())

            if (startLandmark != null && endLandmark != null) {
                val startX = startLandmark.x() * imageWidth * scaleFactor + offsetX
                val startY = startLandmark.y() * imageHeight * scaleFactor + offsetY
                val endX = endLandmark.x() * imageWidth * scaleFactor + offsetX
                val endY = endLandmark.y() * imageHeight * scaleFactor + offsetY

                canvas.drawLine(startX, startY, endX, endY, linePaint)
            }
        }
    }

    private fun drawJokerMask(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        val leftEye = faceLandmarks.getOrNull(33) // Left eye landmark index
        val rightEye = faceLandmarks.getOrNull(263) // Right eye landmark index
        val chin = faceLandmarks.getOrNull(152) // Chin landmark index

        if (leftEye != null && rightEye != null && chin != null && jokerMaskBitmap != null) {
            val lx = leftEye.x() * imageWidth * scaleFactor + offsetX
            val ly = leftEye.y() * imageHeight * scaleFactor + offsetY
            val rx = rightEye.x() * imageWidth * scaleFactor + offsetX
            val ry = rightEye.y() * imageHeight * scaleFactor + offsetY
            val cx = chin.x() * imageWidth * scaleFactor + offsetX
            val cy = chin.y() * imageHeight * scaleFactor + offsetY

            // Calculate mask width, height, and rotation
            val maskWidth = Math.hypot((rx - lx).toDouble(), (ry - ly).toDouble()) * 2.0
            val maskHeight = Math.abs(cy - ((ly + ry) / 2)) * 2.0

            val centerX = (lx + rx) / 2
            val centerY = (ly + ry) / 2

            val angle = Math.toDegrees(Math.atan2((ry - ly).toDouble(), (rx - lx).toDouble())).toFloat()

            val maskRect = RectF(
                (centerX - maskWidth / 2).toFloat(),
                (centerY - maskHeight / 3).toFloat(),
                (centerX + maskWidth / 2).toFloat(),
                (centerY + maskHeight * 2 / 3).toFloat()
            )

            val saveCount = canvas.save()
            canvas.rotate(angle, centerX, centerY)
            jokerMaskBitmap?.let {
                canvas.drawBitmap(
                    Bitmap.createScaledBitmap(it, maskRect.width().toInt(), maskRect.height().toInt(), true),
                    maskRect.left,
                    maskRect.top,
                    null
                )
            }
            canvas.restoreToCount(saveCount)
        }
    }
    
    private fun drawGlasses(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        val leftEye = faceLandmarks.getOrNull(33) // Left eye landmark index
        val rightEye = faceLandmarks.getOrNull(263) // Right eye landmark index

        if (leftEye != null && rightEye != null && glassesBitmap != null) {
            val lx = leftEye.x() * imageWidth * scaleFactor + offsetX
            val ly = leftEye.y() * imageHeight * scaleFactor + offsetY
            val rx = rightEye.x() * imageWidth * scaleFactor + offsetX
            val ry = rightEye.y() * imageHeight * scaleFactor + offsetY

            // Calculate glasses width and position
            val glassesWidth = Math.hypot((rx - lx).toDouble(), (ry - ly).toDouble()) * 2.2
            val glassesHeight = glassesWidth * 0.4 // Maintain aspect ratio

            val centerX = (lx + rx) / 2
            val centerY = (ly + ry) / 2

            val angle = Math.toDegrees(Math.atan2((ry - ly).toDouble(), (rx - lx).toDouble())).toFloat()

            val glassesRect = RectF(
                (centerX - glassesWidth / 2).toFloat(),
                (centerY - glassesHeight / 2).toFloat(),
                (centerX + glassesWidth / 2).toFloat(),
                (centerY + glassesHeight / 2).toFloat()
            )

            val saveCount = canvas.save()
            canvas.rotate(angle, centerX, centerY)
            glassesBitmap?.let {
                Log.d(TAG, "Drawing glasses at: ${glassesRect.left}, ${glassesRect.top}, width: ${glassesRect.width()}, height: ${glassesRect.height()}")
                canvas.drawBitmap(
                    Bitmap.createScaledBitmap(it, glassesRect.width().toInt(), glassesRect.height().toInt(), true),
                    glassesRect.left,
                    glassesRect.top,
                    null
                )
            }
            canvas.restoreToCount(saveCount)
        } else {
            Log.d(TAG, "Cannot draw glasses: leftEye=${leftEye != null}, rightEye=${rightEye != null}, glassesBitmap=${glassesBitmap != null}")
        }
    }

    fun setResults(
        faceLandmarkerResults: FaceLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = faceLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 8F
        private const val TAG = "Face Landmarker Overlay"
    }
}
