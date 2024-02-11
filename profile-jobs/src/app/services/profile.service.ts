import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { Profile } from '../interface/profile';
import {environment} from "../../environments/environment.prod";


@Injectable({
  providedIn: 'root'
})
export class ProfileService {

  private apiUrl = environment.url;

  constructor(private http: HttpClient) {}

  checkServerStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/info`);
  }

  getProfileClassifier(profile: Profile): Observable<any> {
    return this.http.post(`${this.apiUrl}/classify_profile`, profile);
  }

}
