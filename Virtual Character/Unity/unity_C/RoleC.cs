using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//TCP
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Linq;

public class RoleC : MonoBehaviour
{
    Thread receiveThread;
    TcpClient client;
    TcpListener listener;

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
							var incommingData=new byte[length];
							Array.Copy(bytes,0,incommingData,0,length);
							string clientMessage=Encoding.ASCII.GetString(incommingData);
							string[] res=clientMessage.Split(' ');
							
							//LeftArm
							leftshoulder_position[0]=(float)Convert.ToDouble(res[0]); leftshoulder_position[1]=(float)Convert.ToDouble(res[1]); leftshoulder_position[2]=(float)Convert.ToDouble(res[2]);
							leftelbow_position[0]=(float)Convert.ToDouble(res[3]); leftelbow_position[1]=(float)Convert.ToDouble(res[4]); leftelbow_position[2]=(float)Convert.ToDouble(res[5]);
							lefthand_position[0]=(float)Convert.ToDouble(res[6]); lefthand_position[1]=(float)Convert.ToDouble(res[7]); lefthand_position[2]=(float)Convert.ToDouble(res[8]);
							
							//Neck
							neck_position[0]=(float)Convert.ToDouble(res[9]); neck_position[1]=(float)Convert.ToDouble(res[10]); neck_position[2]=(float)Convert.ToDouble(res[11]);
							
							//RightArm
							rightshoulder_position[0]=(float)Convert.ToDouble(res[12]); rightshoulder_position[1]=(float)Convert.ToDouble(res[13]); rightshoulder_position[2]=(float)Convert.ToDouble(res[14]);
							rightelbow_position[0]=(float)Convert.ToDouble(res[15]); rightelbow_position[1]=(float)Convert.ToDouble(res[16]); rightelbow_position[2]=(float)Convert.ToDouble(res[17]);
							righthand_position[0]=(float)Convert.ToDouble(res[18]); righthand_position[1]=(float)Convert.ToDouble(res[19]); righthand_position[2]=(float)Convert.ToDouble(res[20]);
							
							
							
							//LeftThigh
							leftupleg_position[0]=(float)Convert.ToDouble(res[21]); leftupleg_position[1]=(float)Convert.ToDouble(res[22]); leftupleg_position[2]=(float)Convert.ToDouble(res[23]);
							leftleg_position[0]=(float)Convert.ToDouble(res[24]); leftleg_position[1]=(float)Convert.ToDouble(res[25]); leftleg_position[2]=(float)Convert.ToDouble(res[26]);
							leftfoot_position[0]=(float)Convert.ToDouble(res[27]); leftfoot_position[1]=(float)Convert.ToDouble(res[28]); leftfoot_position[2]=(float)Convert.ToDouble(res[29]);
							
							//RightThigh
							rightupleg_position[0]=(float)Convert.ToDouble(res[30]); rightupleg_position[1]=(float)Convert.ToDouble(res[31]); rightupleg_position[2]=(float)Convert.ToDouble(res[32]);
							rightleg_position[0]=(float)Convert.ToDouble(res[33]); rightleg_position[1]=(float)Convert.ToDouble(res[34]); rightleg_position[2]=(float)Convert.ToDouble(res[35]);
							rightfoot_position[0]=(float)Convert.ToDouble(res[36]); rightfoot_position[1]=(float)Convert.ToDouble(res[37]); rightfoot_position[2]=(float)Convert.ToDouble(res[38]);
							
							Debug.Log("ok");
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
    

    //public Animator anim;
    //public SkinnedMeshRenderer eye,eye_lid,mouth,eyebrow;
    
    public Transform test1,test2;
    
    //LeftArm
    public Transform leftshoulder,leftelbow,lefthand,
    leftshoulder_now,leftelbow_now,lefthand_now;
    Vector3 leftshoulder_position,leftelbow_position,lefthand_position;
    
    //Neck
    public Transform neck,
	neck_now;
    Vector3 neck_position;
    Quaternion neck_quat;
    
    //RightArm
    public Transform rightshoulder,rightelbow,righthand,
    rightshoulder_now,rightelbow_now,righthand_now;
    Vector3 rightshoulder_position,rightelbow_position,righthand_position;
    
    
    
    //LeftThigh
    public Transform leftupleg,leftleg,leftfoot,
    leftupleg_now,leftleg_now,leftfoot_now;
    Vector3 leftupleg_position,leftleg_position,leftfoot_position;
    
    //RightThigh
    public Transform rightupleg,rightleg,rightfoot,
    rightupleg_now,rightleg_now,rightfoot_now;
    Vector3 rightupleg_position,rightleg_position,rightfoot_position;
    
    
    
    void Start()
    {
    	InitSocket();
		Debug.Log("test123");
        
        neck_quat=Quaternion.Euler(0,-90,-90);
        
        /*
        anim=GetComponent<Animator>();
        neck=anim.GetBoneTransform(HumanBodyBones.Neck);
        neck.rotation=Quaternion.Euler(30,0,0);
        */
        
        
        
        /*
        leftshoulder.position=leftshoulder_now.position; leftelbow.position=leftelbow_now.position; lefthand.position=lefthand_now.position;
        neck.position=neck_now.position;
        rightshoulder.position=rightshoulder_now.position; rightelbow.position=rightelbow_now.position; righthand.position=righthand_now.position;
        
        leftupleg.position=leftupleg_now.position; leftleg.position=leftleg_now.position; leftfoot.position=leftfoot_now.position;
        rightupleg.position=rightupleg_now.position; rightleg.position=rightleg_now.position; rightfoot.position=rightfoot_now.position;
        */
        
        
        leftshoulder.position=new Vector3(-0.1158f,1.2298f,-0.0098f);
    }

	float a=0;
    void Update()
    {
		if(Input.GetKeyDown(KeyCode.Space)) {transform.Translate(0,0,1*Time.deltaTime);}
		if(Input.GetKey(KeyCode.A))
		{
			a+=1;
			neck.rotation=Quaternion.Euler(0,a,0)*neck_quat;
			righthand.position=new Vector3(a,a,a);
			Debug.Log(a);
		}
		
		if(Input.GetKey("up")) {transform.Translate(0,0,0.05f);}
		if(Input.GetKey("down")) {transform.Translate(0,0,-0.05f);}
	 	if(Input.GetKey("left")) {transform.Rotate(0,-3,0);}
		if(Input.GetKey("right")) {transform.Rotate(0,3,0);}
		
        
        //test1.position=new Vector3(-leftshoulder_x+0.5f,-leftshoulder_y+1.4f,-leftshoulder_z/2f-0.25f);
        //test2.position=new Vector3(-leftelbow_x+0.5f,-leftelbow_y+1.4f,-leftelbow_z/2f-0.25f);

        //test1.rotation=Quaternion.LookRotation(leftelbow_position-leftshoulder_position);

        
        /*
		//LeftArm
		leftshoulder.position=new Vector3(-leftshoulder_position[0]+0.5f,-leftshoulder_position[1]+1.4f,-leftshoulder_position[2]/2f-0.25f);
		leftelbow.position=new Vector3(-leftelbow_position[0]+0.5f,-leftelbow_position[1]+1.4f,-leftelbow_position[2]/2f-0.25f);
		lefthand.position=new Vector3(-lefthand_position[0]+0.5f,-lefthand_position[1]+1.4f,-lefthand_position[2]/2f-0.25f);
		
		//Neck
		neck.position=new Vector3(-neck_position[0]+0.5f,-neck_position[1]+1.4f,-neck_position[2]/2f-0.25f);
		
		//RightArm
		rightshoulder.position=new Vector3(-rightshoulder_position[0]+0.5f,-rightshoulder_position[1]+1.4f,-rightshoulder_position[2]/2f-0.25f);
		rightelbow.position=new Vector3(-rightelbow_position[0]+0.5f,-rightelbow_position[1]+1.4f,-rightelbow_position[2]/2f-0.25f);
		righthand.position=new Vector3(-righthand_position[0]+0.5f,-righthand_position[1]+1.4f,-righthand_position[2]/2f-0.25f);
		
		
		
		//LeftThigh
		leftupleg.position=new Vector3(-leftupleg_position[0]+0.5f,-leftupleg_position[1]+1.4f,-leftupleg_position[2]/2f-0.25f);
		leftleg.position=new Vector3(-leftleg_position[0]+0.5f,-leftleg_position[1]+1.4f,-leftleg_position[2]/2f-0.25f);
		leftfoot.position=new Vector3(-leftfoot_position[0]+0.5f,-leftfoot_position[1]+1.4f,-leftfoot_position[2]/2f-0.25f);
		
		//RightThigh
		rightupleg.position=new Vector3(-rightupleg_position[0]+0.5f,-rightupleg_position[1]+1.4f,-rightupleg_position[2]/2f-0.25f);
		rightleg.position=new Vector3(-rightleg_position[0]+0.5f,-rightleg_position[1]+1.4f,-rightleg_position[2]/2f-0.25f);
		rightfoot.position=new Vector3(-rightfoot_position[0]+0.5f,-rightfoot_position[1]+1.4f,-rightfoot_position[2]/2f-0.25f);
        */
        
        
        //test1.position=lefthand_position;
        
        //LeftArm
        leftshoulder.rotation=Quaternion.LookRotation(leftelbow_position-leftshoulder_position)*Quaternion.Euler(0,-90,0);
        leftelbow.rotation=Quaternion.LookRotation(new Vector3(lefthand_position[0]-leftelbow_position[0],lefthand_position[2]-leftelbow_position[2],lefthand_position[1]-leftelbow_position[1]))*Quaternion.Euler(-90,0,0);
        //lefthand.rotation=Quaternion.LookRotation(test1.position-lefthand_position);
    }
}
