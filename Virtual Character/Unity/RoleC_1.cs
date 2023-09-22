using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Linq;


public class RoleC_1 : MonoBehaviour
{
    Thread receiveThread;
    TcpClient client;
    TcpListener listener;

    
    
    int actions,fire_time;
    float speed = 8f;
    bool fire=false,continue_fire=false,connected;
    public Animator anim;
    public Transform test1,test2;
    public GameObject cam1,cam2,head_set,bullet,bullet_copy;
    
    //public Transform[] cube_pose = new Transform[15];
    
    //Hips
    Vector3 hips_position;
    Transform hips_t;
    new Vector3 hips1_rotation,hips2_rotation;
    
    //Neck RollPitchYaw
    new Vector3 RPY_rotation;
    
    //Mouth
    Vector3 leftmouth_position,rightmouth_position;
    Transform leftmouth_t,rightmouth_t;
    new Vector3 mouth_rotation;
    
    //LeftArm
    Vector3 leftarm_position,leftforearm_position,lefthand_position,leftindex_position,leftpinky_position;
    Transform leftarm_t,leftforearm_t,lefthand_t,leftindex_t,leftpinky_t;
    new Vector3 lefthand1_rotation,lefthand2_rotation;
    
    //LeftFinger
    Vector3[] leftfinger_position = new Vector3[21];
    Transform[] leftfinger_t = new Transform[15];
    
    //Spine
    Vector3 spine_position,neck_position,head_position;
    Transform spine_t,neck_t,head_t;
    Quaternion neck_quat;
    
    //RightArm
    Vector3 rightarm_position,rightforearm_position,righthand_position,rightindex_position,rightpinky_position;
    Transform rightarm_t,rightforearm_t,righthand_t,rightindex_t,rightpinky_t;
    
    
    
    //LeftThigh
    Vector3 leftupleg_position,leftleg_position,leftfoot_position;
    Transform leftupleg_t,leftleg_t,leftfoot_t;
    
    //RightThigh
    Vector3 rightupleg_position,rightleg_position,rightfoot_position;
    Transform rightupleg_t,rightleg_t,rightfoot_t;
        
    
    void InitSocket()
    {
        receiveThread=new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

	void ReceiveData()
	{
		try
		{
			listener=new TcpListener(IPAddress.Parse("127.0.12.12"),1224);
			listener.Start();
			byte[] bytes=new byte[1024];
			while(true)
			{
				using(client=listener.AcceptTcpClient())
				{
					using(NetworkStream stream=client.GetStream())
					{
						int length;
						while((length=stream.Read(bytes,0,bytes.Length))!=0)
						{
                            try
                            {
                                var incommingData=new byte[length];
                                Array.Copy(bytes,0,incommingData,0,length);
                                string clientMessage=Encoding.ASCII.GetString(incommingData);
                                string[] res=clientMessage.Split(' ');

                                //LeftArm
                                leftarm_position[0]=(float)Convert.ToDouble(res[0]); leftarm_position[1]=(float)Convert.ToDouble(res[1]); leftarm_position[2]=(float)Convert.ToDouble(res[2]);
                                leftforearm_position[0]=(float)Convert.ToDouble(res[3]); leftforearm_position[1]=(float)Convert.ToDouble(res[4]); leftforearm_position[2]=(float)Convert.ToDouble(res[5]);
                                lefthand_position[0]=(float)Convert.ToDouble(res[6]); lefthand_position[1]=(float)Convert.ToDouble(res[7]); lefthand_position[2]=(float)Convert.ToDouble(res[8]);

                                //Neck
                                neck_position[0]=(float)Convert.ToDouble(res[9]); neck_position[1]=(float)Convert.ToDouble(res[10]); neck_position[2]=(float)Convert.ToDouble(res[11]);

                                //RightArm
                                rightarm_position[0]=(float)Convert.ToDouble(res[12]); rightarm_position[1]=(float)Convert.ToDouble(res[13]); rightarm_position[2]=(float)Convert.ToDouble(res[14]);
                                rightforearm_position[0]=(float)Convert.ToDouble(res[15]); rightforearm_position[1]=(float)Convert.ToDouble(res[16]); rightforearm_position[2]=(float)Convert.ToDouble(res[17]);
                                righthand_position[0]=(float)Convert.ToDouble(res[18]); righthand_position[1]=(float)Convert.ToDouble(res[19]); righthand_position[2]=(float)Convert.ToDouble(res[20]);



                                //LeftThigh
                                leftupleg_position[0]=(float)Convert.ToDouble(res[21]); leftupleg_position[1]=(float)Convert.ToDouble(res[22]); leftupleg_position[2]=(float)Convert.ToDouble(res[23]);
                                leftleg_position[0]=(float)Convert.ToDouble(res[24]); leftleg_position[1]=(float)Convert.ToDouble(res[25]); leftleg_position[2]=(float)Convert.ToDouble(res[26]);
                                leftfoot_position[0]=(float)Convert.ToDouble(res[27]); leftfoot_position[1]=(float)Convert.ToDouble(res[28]); leftfoot_position[2]=(float)Convert.ToDouble(res[29]);

                                //RightThigh
                                rightupleg_position[0]=(float)Convert.ToDouble(res[30]); rightupleg_position[1]=(float)Convert.ToDouble(res[31]); rightupleg_position[2]=(float)Convert.ToDouble(res[32]);
                                rightleg_position[0]=(float)Convert.ToDouble(res[33]); rightleg_position[1]=(float)Convert.ToDouble(res[34]); rightleg_position[2]=(float)Convert.ToDouble(res[35]);
                                rightfoot_position[0]=(float)Convert.ToDouble(res[36]); rightfoot_position[1]=(float)Convert.ToDouble(res[37]); rightfoot_position[2]=(float)Convert.ToDouble(res[38]);

                                //Hips
                                hips_position[0]=(float)Convert.ToDouble(res[39]); hips_position[1]=(float)Convert.ToDouble(res[40]); hips_position[2]=(float)Convert.ToDouble(res[41]);

                                //Mouth
                                leftmouth_position[0]=(float)Convert.ToDouble(res[42]); leftmouth_position[1]=(float)Convert.ToDouble(res[43]); leftmouth_position[2]=(float)Convert.ToDouble(res[44]);
                                rightmouth_position[0]=(float)Convert.ToDouble(res[45]); rightmouth_position[1]=(float)Convert.ToDouble(res[46]); rightmouth_position[2]=(float)Convert.ToDouble(res[47]);

                                //Neck RollPitchYaw
                                RPY_rotation[0]=(float)Convert.ToDouble(res[48]); RPY_rotation[1]=(float)Convert.ToDouble(res[49]); RPY_rotation[2]=(float)Convert.ToDouble(res[50]);
                                
                                //Action
                                actions=(int)Convert.ToDouble(res[51]);

                                
                                
                                for(int i=0;i<21;i++) for(int j=0;j<3;j++) leftfinger_position[i][j] =(float)Convert.ToDouble(res[i*3+j+52]);
                                
                                /*
                                //LeftHand
                                leftindex_position[0]=(float)Convert.ToDouble(res[52]); leftindex_position[1]=(float)Convert.ToDouble(res[53]); leftindex_position[2]=(float)Convert.ToDouble(res[54]);
                                leftpinky_position[0]=(float)Convert.ToDouble(res[55]); leftpinky_position[1]=(float)Convert.ToDouble(res[56]); leftpinky_position[2]=(float)Convert.ToDouble(res[57]);
                                
                                //RightHand
                                rightindex_position[0]=(float)Convert.ToDouble(res[47]); rightindex_position[1]=(float)Convert.ToDouble(res[48]); rightindex_position[2]=(float)Convert.ToDouble(res[49]);
                                rightpinky_position[0]=(float)Convert.ToDouble(res[50]); rightpinky_position[1]=(float)Convert.ToDouble(res[51]); rightpinky_position[2]=(float)Convert.ToDouble(res[52]);
                                */

                                
                                connected=true;
                                Debug.Log("connected");
                            }
                            
                            catch(Exception)
                            {
                                connected=false;
                                Debug.Log("not connected");
                                continue;
                            }
						}
					}
				}
			}
		}
		catch(Exception e)
		{
			Debug.Log(e.ToString()); 
		}
	}
    
    
    
    void Start()
    {
        Debug.Log("test123");
    	InitSocket();
        anim=GetComponent<Animator>();
        
        hips_t=anim.GetBoneTransform(HumanBodyBones.Hips);
        
        spine_t=anim.GetBoneTransform(HumanBodyBones.Spine);
        neck_t=anim.GetBoneTransform(HumanBodyBones.Neck);
        head_t=anim.GetBoneTransform(HumanBodyBones.Head);
        
        leftarm_t=anim.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        leftforearm_t=anim.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        lefthand_t=anim.GetBoneTransform(HumanBodyBones.LeftHand);
        
        leftfinger_t[0]=anim.GetBoneTransform(HumanBodyBones.LeftThumbProximal);
        leftfinger_t[1]=anim.GetBoneTransform(HumanBodyBones.LeftThumbIntermediate);
        leftfinger_t[2]=anim.GetBoneTransform(HumanBodyBones.LeftThumbDistal);
        
        leftfinger_t[3]=anim.GetBoneTransform(HumanBodyBones.LeftIndexProximal);
        leftfinger_t[4]=anim.GetBoneTransform(HumanBodyBones.LeftIndexIntermediate);
        leftfinger_t[5]=anim.GetBoneTransform(HumanBodyBones.LeftIndexDistal);
        
        leftfinger_t[6]=anim.GetBoneTransform(HumanBodyBones.LeftMiddleProximal);
        leftfinger_t[7]=anim.GetBoneTransform(HumanBodyBones.LeftMiddleIntermediate);
        leftfinger_t[8]=anim.GetBoneTransform(HumanBodyBones.LeftMiddleDistal);
        
        leftfinger_t[9]=anim.GetBoneTransform(HumanBodyBones.LeftRingProximal);
        leftfinger_t[10]=anim.GetBoneTransform(HumanBodyBones.LeftRingIntermediate);
        leftfinger_t[11]=anim.GetBoneTransform(HumanBodyBones.LeftRingDistal);
        
        leftfinger_t[12]=anim.GetBoneTransform(HumanBodyBones.LeftLittleProximal);
        leftfinger_t[13]=anim.GetBoneTransform(HumanBodyBones.LeftLittleIntermediate);
        leftfinger_t[14]=anim.GetBoneTransform(HumanBodyBones.LeftLittleDistal);
        
        
        
        rightarm_t=anim.GetBoneTransform(HumanBodyBones.RightUpperArm);
        rightforearm_t=anim.GetBoneTransform(HumanBodyBones.RightLowerArm);
        righthand_t=anim.GetBoneTransform(HumanBodyBones.RightHand);
        
        leftupleg_t=anim.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        leftleg_t=anim.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        leftfoot_t=anim.GetBoneTransform(HumanBodyBones.LeftFoot);
        
        rightupleg_t=anim.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        rightleg_t=anim.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        rightfoot_t=anim.GetBoneTransform(HumanBodyBones.RightFoot);
        
        
        
        cam1.SetActive(true);
        cam2.SetActive(false);
        bullet.SetActive(false);
    }

	//float a=0;
    void Update()
    {
        /*
		if(Input.GetKeyDown(KeyCode.Space)) {transform.Translate(0,0,1*Time.deltaTime);}
		if(Input.GetKey(KeyCode.A))
		{
			a+=1;
			//neck_t.rotation=Quaternion.Euler(0,a,0);
            //neck_t.eulerAngles=new Vector3(0,a,0);
            
            neck_t.eulerAngles=Quaternion.Euler(0,a,0).ToEulerAngles()*120;
			Debug.Log(a);
		}
		
		if(Input.GetKey("up")) {transform.Translate(0,0,0.05f);}
		if(Input.GetKey("down")) {transform.Translate(0,0,-0.05f);}
	 	if(Input.GetKey("left")) {transform.Rotate(0,-3,0);}
		if(Input.GetKey("right")) {transform.Rotate(0,3,0);}
        */
        
        
		
        if(connected)
        {
            /*
            //Hips
            hips_t.position=new Vector3(-hips_position.x*8,-hips_position.y*8,-hips_position.z*7);
            //hips_t.rotation=Quaternion.LookRotation((rightarm_position+rightupleg_position)/2-(leftarm_position+leftupleg_position)/2)*Quaternion.Euler(0,0,-90);

            hips1_rotation=Quaternion.LookRotation((rightarm_position+rightupleg_position)/2-(leftarm_position+leftupleg_position)/2).ToEulerAngles()*60;
            hips2_rotation=Quaternion.LookRotation((rightarm_position+leftarm_position)/2-(rightupleg_position+leftupleg_position)/2).ToEulerAngles()*60;
            hips_t.eulerAngles=new Vector3(hips1_rotation.x,hips1_rotation.y,hips2_rotation.x+180);

            //LeftArm
            leftarm_t.rotation=Quaternion.LookRotation(leftforearm_position-leftarm_position)*Quaternion.Euler(90,-90,0);
            leftforearm_t.rotation=Quaternion.LookRotation(lefthand_position-leftforearm_position)*Quaternion.Euler(90,-90,0);
            //lefthand_t.rotation=Quaternion.LookRotation(leftindex_position-leftpinky_position)*Quaternion.Euler(0,90,-90);

            //Neck
            //neck_t.rotation=Quaternion.LookRotation(leftmouth_position-rightmouth_position)*Quaternion.Euler(0,0,-90);
            neck_t.rotation=Quaternion.Euler(-RPY_rotation*2)*Quaternion.Euler(0,-90,-90);

            //mouth_rotation=Quaternion.LookRotation(leftmouth_position-rightmouth_position).ToEulerAngles()*60;
            //neck_t.eulerAngles=new Vector3(-mouth_rotation.y+90,-mouth_rotation.x+90,-mouth_rotation.z+90);

            //RightArm
            rightarm_t.rotation=Quaternion.LookRotation(rightforearm_position-rightarm_position)*Quaternion.Euler(-90,-90,0);
            rightforearm_t.rotation=Quaternion.LookRotation(righthand_position-rightforearm_position)*Quaternion.Euler(-90,-90,0);
            //righthand_t.rotation=Quaternion.LookRotation(rightindex_position-rightpinky_position)*Quaternion.Euler(90,0,90);



            //LeftThigh
            leftupleg_t.rotation=Quaternion.LookRotation(leftleg_position-leftupleg_position)*Quaternion.Euler(0,-90,0);
            leftleg_t.rotation=Quaternion.LookRotation(leftfoot_position-leftleg_position)*Quaternion.Euler(180,-90,-45);

            //RightThigh
            rightupleg_t.rotation=Quaternion.LookRotation(rightleg_position-rightupleg_position)*Quaternion.Euler(0,-90,0);
            rightleg_t.rotation=Quaternion.LookRotation(rightfoot_position-rightleg_position)*Quaternion.Euler(180,-90,-45);

            //test1.position=leftindex_position;
            //test2.position=leftpinky_position;
            */
            
            
            
            
            hips_t.position=Vector3.Slerp(hips_t.position,new Vector3(-hips_position.x*8,-hips_position.y*8,-hips_position.z*7),speed*Time.deltaTime);
            hips_t.position=Vector3.Slerp(hips_t.position,new Vector3(-hips_position.x*8,-1.8f,-hips_position.z*7),speed*Time.deltaTime);
            
            hips1_rotation=Quaternion.LookRotation((rightarm_position+rightupleg_position)/2-(leftarm_position+leftupleg_position)/2).ToEulerAngles()*60;
            hips2_rotation=Quaternion.LookRotation((rightarm_position+leftarm_position)/2-(rightupleg_position+leftupleg_position)/2).ToEulerAngles()*60;
            hips_t.rotation=Quaternion.Slerp(hips_t.rotation,Quaternion.Euler(new Vector3(hips1_rotation.x,hips1_rotation.y,hips2_rotation.x+180)),speed*Time.deltaTime);
            
            //LeftArm
            leftarm_t.rotation=Quaternion.Slerp(leftarm_t.rotation,Quaternion.LookRotation(leftforearm_position-leftarm_position)*Quaternion.Euler(90,-90,0),speed*Time.deltaTime);
            leftforearm_t.rotation=Quaternion.Slerp(leftforearm_t.rotation,Quaternion.LookRotation(lefthand_position-leftforearm_position)*Quaternion.Euler(90,-90,0),speed*Time.deltaTime);
            
            //LeftHand
            lefthand1_rotation=Quaternion.LookRotation(leftfinger_position[9]-leftfinger_position[0]).ToEulerAngles()*60;
            lefthand2_rotation=Quaternion.LookRotation(leftfinger_position[5]-leftfinger_position[17]).ToEulerAngles()*60;
            ///lefthand_t.rotation=Quaternion.Slerp(lefthand_t.rotation,Quaternion.Euler(new Vector3(-lefthand1_rotation.x+90,lefthand2_rotation.y+90,-lefthand1_rotation.y+90)),speed*Time.deltaTime);
            //lefthand_t.rotation=Quaternion.Slerp(lefthand_t.rotation,Quaternion.Euler(new Vector3(-lefthand1_rotation.x+90,lefthand2_rotation.y+90,-lefthand1_rotation.y+90))*Quaternion.Euler(180,180,180),speed*Time.deltaTime);
                
            lefthand_t.rotation=Quaternion.Slerp(lefthand_t.rotation,Quaternion.LookRotation(leftfinger_position[9]-leftfinger_position[0])*Quaternion.Euler(90,-90,0),speed*Time.deltaTime);
            lefthand_t.rotation=Quaternion.Slerp(lefthand_t.rotation,Quaternion.Euler(new Vector3(lefthand_t.rotation.x+90,lefthand2_rotation.y+180,lefthand_t.rotation.z)),speed*Time.deltaTime);
            
            //lefthand_t.rotation=Quaternion.Slerp(lefthand_t.rotation,Quaternion.Euler(new Vector3(-lefthand1_rotation.y,-lefthand1_rotation.z,lefthand_t.position.z))*Quaternion.Euler(0,90,90),speed*Time.deltaTime);
            
            //LeftFinger
            for(int left_i=1;left_i<3;left_i++) leftfinger_t[left_i].rotation=Quaternion.Slerp(leftfinger_t[left_i].rotation,Quaternion.LookRotation(leftfinger_position[left_i+2]-leftfinger_position[left_i+1])*Quaternion.Euler(90,-90,0),speed*Time.deltaTime);
            
            for(int left_i=0;left_i<3;left_i++) leftfinger_t[left_i+3].rotation=Quaternion.Slerp(leftfinger_t[left_i+3].rotation,Quaternion.LookRotation(leftfinger_position[left_i+6]-leftfinger_position[left_i+5])*Quaternion.Euler(90,-90,0),speed*Time.deltaTime);
            
            for(int left_i=0;left_i<3;left_i++) leftfinger_t[left_i+6].rotation=Quaternion.Slerp(leftfinger_t[left_i+6].rotation,Quaternion.LookRotation(leftfinger_position[left_i+10]-leftfinger_position[left_i+9])*Quaternion.Euler(90,-90,0),speed*Time.deltaTime);
            
            for(int left_i=0;left_i<3;left_i++) leftfinger_t[left_i+9].rotation=Quaternion.Slerp(leftfinger_t[left_i+9].rotation,Quaternion.LookRotation(leftfinger_position[left_i+14]-leftfinger_position[left_i+13])*Quaternion.Euler(90,-90,0),speed*Time.deltaTime);
            
            for(int left_i=0;left_i<3;left_i++) leftfinger_t[left_i+12].rotation=Quaternion.Slerp(leftfinger_t[left_i+12].rotation,Quaternion.LookRotation(leftfinger_position[left_i+18]-leftfinger_position[left_i+17])*Quaternion.Euler(90,-90,0),speed*Time.deltaTime);
            
            //test1.position=leftfinger_position[1];
            //test2.position=leftfinger_position[2];
            

            //Neck
            neck_t.rotation=Quaternion.Slerp(neck_t.rotation,Quaternion.Euler(-RPY_rotation*2)*Quaternion.Euler(0,-90,-90),speed*Time.deltaTime);

            //RightArm
            rightarm_t.rotation=Quaternion.Slerp(rightarm_t.rotation,Quaternion.LookRotation(rightforearm_position-rightarm_position)*Quaternion.Euler(-90,-90,0),speed*Time.deltaTime);
            rightforearm_t.rotation=Quaternion.Slerp(rightforearm_t.rotation,Quaternion.LookRotation(righthand_position-rightforearm_position)*Quaternion.Euler(-90,-90,0),speed*Time.deltaTime);



            //LeftThigh
            leftupleg_t.rotation=Quaternion.Slerp(leftupleg_t.rotation,Quaternion.LookRotation(leftleg_position-leftupleg_position)*Quaternion.Euler(0,-90,0),speed*Time.deltaTime);
            leftleg_t.rotation=Quaternion.Slerp(leftleg_t.rotation,Quaternion.LookRotation(leftfoot_position-leftleg_position)*Quaternion.Euler(180,-90,-45),speed*Time.deltaTime);

            //RightThigh
            rightupleg_t.rotation=Quaternion.Slerp(rightupleg_t.rotation,Quaternion.LookRotation(rightleg_position-rightupleg_position)*Quaternion.Euler(0,-90,0),speed*Time.deltaTime);
            rightleg_t.rotation=Quaternion.Slerp(rightleg_t.rotation,Quaternion.LookRotation(rightfoot_position-rightleg_position)*Quaternion.Euler(180,-90,-45),speed*Time.deltaTime);
            
            
            /*
            //============================================================================================
            cube_pose[0].position=-leftarm_position;
            cube_pose[1].position=-leftforearm_position;
            cube_pose[2].position=-lefthand_position;
            cube_pose[3].position=-rightarm_position;
            cube_pose[4].position=-rightforearm_position;
            cube_pose[5].position=-righthand_position;
            cube_pose[6].position=-leftupleg_position;
            cube_pose[7].position=-leftleg_position;
            cube_pose[8].position=-leftfoot_position;
            cube_pose[9].position=-rightupleg_position;
            cube_pose[10].position=-rightleg_position;
            cube_pose[11].position=-rightfoot_position;
            //cube_pose[12].position=leftarm_position;
            //cube_pose[13].position=leftarm_position;
            //cube_pose[14].position=leftarm_position;
            //============================================================================================
            */
            
            
            if(actions==1 && !continue_fire && fire_time>1000)
            {
                continue_fire=true;
                fire_time=0;
                bullet_copy=Instantiate(this.bullet);
                bullet_copy.SetActive(true);
                bullet_copy.transform.forward=leftforearm_t.position-lefthand_t.position;
            }
            else if(actions==1 && fire_time>1000)
            {
                continue_fire=false;
                fire_time=0;
                Destroy(bullet_copy);
            }
            fire_time++;
        }
        
        
        
        cam2.transform.rotation=head_t.rotation*Quaternion.Euler(-70,90,0);
        cam2.transform.position=head_t.position+new Vector3(0,0.1f,0.2f);
        
        if(Input.GetKey(KeyCode.Alpha1))
        {
            cam1.SetActive(true);
            cam2.SetActive(false);
            head_set.SetActive(true);
        }
        if(Input.GetKey(KeyCode.Alpha2))
        {
            cam1.SetActive(false);
            cam2.SetActive(true);
            head_set.SetActive(false);
        }
        
        
        
        bullet.transform.rotation=lefthand_t.rotation*Quaternion.Euler(0,90,0);
        bullet.transform.position=lefthand_t.position;
        
        if(!fire)
        {
            if(Input.GetKey(KeyCode.Mouse0))
            {
                bullet_copy=Instantiate(this.bullet);
                bullet_copy.SetActive(true);
                bullet_copy.transform.forward=leftforearm_t.position-lefthand_t.position;
                fire=true;
            }
        }
        else 
        {
            bullet_copy.transform.position-=bullet_copy.transform.forward*Time.deltaTime*50;
        }
            
        if((bullet_copy.transform.position-lefthand_t.position).magnitude>20 && fire==true)
        {
            Destroy(bullet_copy);
            fire=false;
        }
        
        

        if(continue_fire)
        {
            bullet_copy.transform.position-=bullet_copy.transform.forward*Time.deltaTime*50;
            
            if((bullet_copy.transform.position-lefthand_t.position).magnitude>20)
            {
                Destroy(bullet_copy);
                bullet_copy=Instantiate(this.bullet);
                bullet_copy.SetActive(true);
                bullet_copy.transform.forward=leftforearm_t.position-lefthand_t.position;
            }
        }
    }
}
