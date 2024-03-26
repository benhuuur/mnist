using System.Drawing;

namespace ImageProcessing;

public static partial class ImageGenerator
{
    public static Bitmap GenerateImageFromGrayVector(
        byte[] grayVector,
        int height = 28,
        int width = 28
    )
    {
        int index = 0;
        Color pixelColor;
        Bitmap bitmap = new Bitmap(width, height);
        Graphics graphics = Graphics.FromImage(bitmap);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte intensity = grayVector[index];
                pixelColor = Color.FromArgb(intensity, intensity, intensity);
                index++;
                graphics.FillRectangle(new SolidBrush(pixelColor), x, y, 1, 1);
            }
        }

        bitmap.Save("images/output.png", System.Drawing.Imaging.ImageFormat.Png);
        return bitmap;
    }
}
