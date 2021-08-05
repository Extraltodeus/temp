using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class textureGenerator : MonoBehaviour {

    public static Shader shader1;

    void Start()
    {
        
    }
    

    public static Texture2D pinpinTexture(float scale, float pattern = 0)
    {
        int pixWidth  = 512;
        int pixHeight = 512;
        float xOrg = 2;
        float yOrg = 2;
        float margin = pixWidth/36;

        Texture2D noiseTex = new Texture2D(pixWidth, pixHeight);
        Color[] pix = new Color[noiseTex.width * noiseTex.height];

        float y = 0.0F;
        float ratio = 0;
        while (y < noiseTex.height)
        {
            float x = 0.0F;
            while (x < noiseTex.width)
            {
                float xCoord = pattern + xOrg + x / noiseTex.width  * scale;
                float yCoord = pattern + yOrg + y / noiseTex.height * scale;
                float sample = Mathf.PerlinNoise(xCoord, yCoord);

                if (x < margin)
                {
                    ratio = utilities.remapRange(x, 0, margin, 0, 1);
                    sample = sample * ratio + Mathf.PerlinNoise(0, yCoord) * (1 - ratio);
                } else if (x > pixWidth - margin)
                {
                    ratio = utilities.remapRange(x, pixWidth - margin, pixWidth, 1, 0);
                    sample = sample * ratio + Mathf.PerlinNoise(0, yCoord) * (1 - ratio);
                }

                sample -= 0.5f * (0.5f-sample);

                pix[(int)y * noiseTex.width + (int)x] = new Color(sample, sample, sample);
                x++;
            }
            y++;
        }

        noiseTex.SetPixels(pix);
        noiseTex.Apply();
        SaveTextureToFile(noiseTex, "texture2d.png");
        return noiseTex;
    }

    static void SaveTextureToFile(Texture2D texture, string filename)
    {
        System.IO.File.WriteAllBytes(filename, texture.EncodeToPNG());
    }
}
