## reference
https://github.com/google/grafika
https://source.android.com/devices/graphics/architecture.html
https://www.bigflake.com/mediacodec/

## Questions
1. where mediaplayer display?
2. what is surfaceview? surfaceholder?
3. why use surfaceview and surfaceholder?
4. what the surfaceview background logic?
5. can canvas and surfaceview overlay?

## how buffers of graphical data move through the system

## SurfaceView
Provides a dedicated drawing surface embedded inside of a view hierarchy. You can control the format of this surface and, if you like, its size; the SurfaceView takes care of placing the surface at the correct location on the screen.

window hold SurfaceView, SurfaceView draw surface. Surface is Z ordered. and SurfaceView punches a hole in its window to allow its surface to be displayed.
so a surfaceview to one surface, one window to one surfaceview? (to -> corresponding to?)

surfaceview sibling: button, textview.
surfaceview takes care of surface.
the view hierarchy takes care of Surface and button or textview. so overlay is possible.
SurfaceHolder interface to access the underlying surface.

surfaceview can provide a surface in which a secondary thread can render into the screen.

One of the purposes of this class is to provide a surface in which a secondary thread can render into the screen. If you are going to use it this way, you need to be aware of some threading semantics:

All SurfaceView and SurfaceHolder.Callback methods will be called from the thread running the SurfaceView's window (typically the main thread of the application). They thus need to correctly synchronize with any state that is also touched by the drawing thread.
You must ensure that the drawing thread only touches the underlying Surface while it is valid -- between SurfaceHolder.Callback.surfaceCreated() and SurfaceHolder.Callback.surfaceDestroyed().

<p>
 * This is very similar to PlayMovieActivity, but the output goes to a SurfaceView instead of
 * a TextureView.  There are some important differences:
 * <ul>
 *   <li> TextureViews behave like normal views.  SurfaceViews don't.  A SurfaceView has
 *        a transparent "hole" in the UI through which an independent Surface layer can
 *        be seen.  This Surface is sent directly to the system graphics compositor.
 *   <li> Because the video is being composited with the UI by the system compositor,
 *        rather than the application, it can often be done more efficiently (e.g. using
 *        a hardware composer "overlay").  This can lead to significant battery savings
 *        when playing a long movie.
 *   <li> On the other hand, the TextureView contents can be freely scaled and rotated
 *        with a simple matrix.  The SurfaceView output is limited to scaling, and it's
 *        more awkward to do.
 *   <li> DRM-protected content can't be touched by the app (or even the system compositor).
 *        We have to point the MediaCodec decoder at a Surface that is composited by a
 *        hardware composer overlay.  The only way to do the app side of this is with
 *        SurfaceView.
 * </ul>
 * <p>
 * The MediaCodec decoder requests buffers from the Surface, passing the video dimensions
 * in as arguments.  The Surface provides buffers with a matching size, which means
 * the video data will completely cover the Surface.  As a result, there's no need to
 * use SurfaceHolder#setFixedSize() to set the dimensions.  The hardware scaler will scale
 * the video to match the view size, so if we want to preserve the correct aspect ratio
 * we need to adjust the View layout.  We can use our custom AspectFrameLayout for this.
 * <p>
 * The actual playback of the video -- sending frames to a Surface -- is the same for
 * TextureView and SurfaceView.
 </p>
