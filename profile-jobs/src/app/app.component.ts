import { Component, OnInit } from '@angular/core';
import { Profile } from './interface/profile';
import { ProfileService } from './services/profile.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'profile-jobs';

  status_server = ""
  profileClassfier = ""
  showClassifierProfile: boolean = false

  // Define un objeto que cumple con la interfaz Profile
  profile: Profile = {
    DSA: 0,
    DBMS: 0,
    OS: 0,
    CN: 0,
    Mathmetics: 0,
    Aptitude: 0,
    Comm: 0,
    Problem_Solving: 0,
    Creative: 0,
    Hackathons: 0,
    Skill_1: 0,
    Skill_2: 0,
  };

  ngOnInit(): void {
    this.checkServerStatus();
    // Verificar el estado del servidor cada 10 segundos
    setInterval(() => {
      this.checkServerStatus();
    }, 10000);
  }

  constructor(private profileService: ProfileService){

  }


  // Método llamado cuando cambia el valor del rango
  onRangoChange(event: any, atributo: keyof Profile) {
    this.profile[atributo] = event.target.value;
    // console.log(`Nuevo valor del rango para ${atributo}:`, this.profile[atributo]);
  }

  // Método llamado cuando se selecciona una habilidad
  onSkillSelect(event: any, skillType: string) {
    const selectedSkill = event.target.value; // Obtener el valor seleccionado
    // console.log(`Habilidad ${skillType} seleccionada:`, selectedSkill);  // Hacer algo con el valor seleccionado (por ejemplo, mostrar en la consola)
  }


  checkServerStatus() {
    this.profileService.checkServerStatus().subscribe(
      (response) => {
        this.status_server = "Running 🟢"
      },
      (error) => {
        this.status_server = "Stop 🔴"
      }
    );
  }

  submit(){
    this.profileService.getProfileClassifier(this.profile).subscribe(
      (response)=> {
        // console.log(response.predicted_profile)
        this.showClassifierProfile = true;
        this.profileClassfier = response.predicted_profile[0];
      },
      (err)=> {
        this.showClassifierProfile = false;
        console.error(err)
      }
    )
  }
}

// https://emojipedia.org/brain
