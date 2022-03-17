using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Dynamic scene management
using UnityEngine.SceneManagement;

namespace RPGFlightmare
{
  public class Scenes
  {
    public const string SCENE_WAREHOUSE = "Environments/Warehouse/Scenes/DemoScene";
    public const string SCENE_GARAGE = "Environments/Garage/Scenes/DemoScene";
    public const string SCENE_TUNNELS = "Environments/Tunnels/Scenes/DemoSceneSimple";
    public const string SCENE_NATUREFOREST = "Environments/Forest/Scenes/DemoScene";
    public const string SCENE_INDUSTRIAL = "Environments/Industrial/Scenes/DemoScene";
    public const string SCENE_STADIUM = "Environments/Arena/Scene/Stadium";
    public const string SCENE_EMPTY = "Environments/Empty/Scenes/Empty";
    public const string SCENE_LIFTOFF = "Environments/Liftoff/Scenes/Liftoff";

    //
    public List<string> scenes_list = new List<string>();
    public int default_scene_id;
    public int num_scene; // number of scenes in total
    public Scenes()
    {
      scenes_list.Add(SCENE_INDUSTRIAL);
      scenes_list.Add(SCENE_STADIUM);
      scenes_list.Add(SCENE_EMPTY);
      scenes_list.Add(SCENE_LIFTOFF);
      //scenes_list.Add(SCENE_WAREHOUSE);
      //scenes_list.Add(SCENE_GARAGE); 
      //scenes_list.Add(SCENE_NATUREFOREST);
      //scenes_list.Add(SCENE_TUNNELS);

      default_scene_id = 0;
      num_scene = scenes_list.Count;
    }

  }
  public class sceneSchedule : MonoBehaviour
  {
    public GameObject camera_template; // Main camera
                                       // public GameObject cam_tmp;
    public GameObject[] time_lines;
    public Scenes scenes = new Scenes();
    public void Start()
    {
      loadScene(scenes.default_scene_id, true);
    }
    public void loadScene(int scene_id, bool camera_preview)
    {
      if (scene_id >= 0 && scene_id < scenes.num_scene)
      {
        SceneManager.LoadScene(scenes.scenes_list[scene_id]);
        scenes.default_scene_id = scene_id;
      }
      else
      {
        SceneManager.LoadScene(scenes.scenes_list[scenes.default_scene_id]);
      }

      /*if (!camera_preview)
      {
        foreach (var tl in time_lines)
        {
          tl.SetActive(false);
        }
      }*/
    }

    public void loadIndustrial()
    {
      loadScene(0, true);
    }

    /*
    public void loadWareHouse()
    {
      loadScene(1, true);
    }
    public void loadGarage()
    {
      loadScene(2, true);
    }

    public void loadForest()
    {
      loadScene(3, true);
    }

    public void loadTunnel()
    {
      loadScene(4, true);
    }
    */

    public void loadStadium()
    {
      loadScene(1, true);
    }

    public void loadEmpty()
    {
      loadScene(2, true);
    }

    public void loadLiftoff()
    {
      loadScene(3, true);
    }

  }
}
